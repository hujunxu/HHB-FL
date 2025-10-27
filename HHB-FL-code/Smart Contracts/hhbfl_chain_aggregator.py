# -*- coding: utf-8 -*-
"""
HHB-FL On-chain Aggregation & Verification (Python reference contract)
=====================================================================
功能：
1) 交易/更新校验：签名、轮次新鲜度、nonce 去重、模式与尺寸检查（防重放/伪造/越界）。

注意：
- 该文件可作为 Hyperledger Fabric External Chaincode / Sawtooth Python 合约或“私链网关服务”。
- 这里的“账本”与“状态”使用 KV 存储抽象，本地以 SQLite/JSON 落地，方便跑单测；部署到链上时替换 KV 后端即可。

"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import time, json, hashlib, os, sqlite3
from ecdsa import VerifyingKey, SECP256k1, BadSignatureError
from binascii import unhexlify, hexlify


# ---------------------------
# 简易 KV 账本：SQLite 实现
# ---------------------------
class KVStore:
    def __init__(self, path="hhbfl_ledger.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS kv (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        )""")
        self.conn.commit()

    def get(self, k: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT v FROM kv WHERE k=?", (k,))
        row = cur.fetchone()
        return row[0] if row else None

    def put(self, k: str, v: str):
        cur = self.conn.cursor()
        cur.execute("REPLACE INTO kv (k, v) VALUES (?,?)", (k, v))
        self.conn.commit()


# ---------------------------
# 数据结构
# ---------------------------
@dataclass
class UpdateTx:
    # 元数据
    round: int
    client_id: str
    ts: int
    nonce: str
    schema: str  # "paillier+ckks/v1"
    # 密文
    paillier_n_hex: str  # 2048-bit modulus n (hex)
    paillier_cw_hex: List[str]  # 权重密文数组，每个元素是 hex(0..n^2-1)
    ckks_cb_payload: str  # 偏置密文负载（base16 或 base64 字符串）
    # 可选：样本数、维度等
    num_samples: int
    w_len: int
    b_len: int
    # 签名
    pubkey_hex: str
    sig_hex: str

    def hash_for_sign(self) -> bytes:
        h = hashlib.sha256()
        payload = json.dumps({
            "round": self.round,
            "client_id": self.client_id,
            "ts": self.ts,
            "nonce": self.nonce,
            "schema": self.schema,
            "paillier_n_hex": self.paillier_n_hex,
            "paillier_cw_hex": self.paillier_cw_hex,
            "ckks_cb_payload": self.ckks_cb_payload,
            "num_samples": self.num_samples,
            "w_len": self.w_len,
            "b_len": self.b_len
        }, sort_keys=True).encode()
        h.update(payload)
        return h.digest()


@dataclass
class RoundState:
    round: int
    start_ts: int
    close_ts: Optional[int]
    n_hex: str  # Paillier n (hex)，固定本轮
    n2_hex: str  # n^2 (hex)
    w_len: int
    b_len: int
    # Paillier 权重密文的同态乘积聚合（初始为 1 mod n^2）
    agg_paillier_cw_hex: List[str]
    # CKKS 偏置“加法序列”日志（按提交顺序保留密文载荷）
    ckks_add_log: List[str]
    # 防重放
    used_nonces: Dict[str, bool]
    # 参与者列表（便于审计）
    participants: List[str]
    # 统计
    accepted: int
    rejected: int


# ---------------------------
# 智能合约主体
# ---------------------------
class HHBFLContract:
    FRESH_WINDOW_SEC = 15 * 60  # 时间新鲜度窗口：15分钟，可按需调整
    MAX_W = 200000  # 安全上限，防止滥用
    MAX_B = 200000

    def __init__(self, kv: KVStore):
        self.kv = kv
        self._bootstrap()

    # ---- 账本键名工具 ----
    def _key_round(self, r: int) -> str:
        return f"round::{r}"

    def _key_tx(self, txid: str) -> str:
        return f"tx::{txid}"

    def _key_head(self) -> str:
        return "meta::head_round"

    # ---- 初始化轮次 ----
    def _bootstrap(self):
        if not self.kv.get(self._key_head()):
            self.open_round(
                r=1,
                n_hex="",  # 首轮可由第一笔合法交易带入 n，或预置
                w_len=0,
                b_len=0
            )

    def open_round(self, r: int, n_hex: str, w_len: int, b_len: int):
        # 构建初始 Paillier 累乘器为 1 (mod n^2)
        n = int(n_hex, 16) if n_hex else 0
        n2_hex = hex((n * n) if n > 0 else 0)[2:]
        agg = ["1"] * w_len if w_len > 0 else []
        st = RoundState(
            round=r, start_ts=int(time.time()), close_ts=None,
            n_hex=n_hex, n2_hex=n2_hex, w_len=w_len, b_len=b_len,
            agg_paillier_cw_hex=agg, ckks_add_log=[],
            used_nonces={}, participants=[], accepted=0, rejected=0
        )
        self.kv.put(self._key_round(r), json.dumps(asdict(st)))
        self.kv.put(self._key_head(), str(r))

    # ---- 获取当前轮次 ----
    def head_round(self) -> int:
        return int(self.kv.get(self._key_head()))

    # ---- 提交更新 ----
    def submit_update(self, tx: UpdateTx) -> Tuple[bool, str]:
        now = int(time.time())
        # 1) 时间新鲜度
        if abs(now - tx.ts) > self.FRESH_WINDOW_SEC:
            self._round_stat(tx.round, accepted=False)
            return False, "stale timestamp"

        # 2) 轮次状态
        st = self._load_round(tx.round)
        if not st or st["close_ts"] is not None:
            self._round_stat(tx.round, accepted=False)
            return False, "round not open"

        # 3) nonce 去重（防重放）——论文中“freshness/anti-replay & deduplication”:contentReference[oaicite:7]{index=7}
        if tx.nonce in st["used_nonces"]:
            self._round_stat(tx.round, accepted=False)
            return False, "nonce reused"

        # 4) 模式&尺寸校验
        if tx.schema != "paillier+ckks/v1":
            self._round_stat(tx.round, accepted=False)
            return False, "unsupported schema"
        if tx.w_len <= 0 or tx.w_len > self.MAX_W or tx.b_len > self.MAX_B:
            self._round_stat(tx.round, accepted=False)
            return False, "invalid dimensions"
        if len(tx.paillier_cw_hex) != tx.w_len:
            self._round_stat(tx.round, accepted=False)
            return False, "weight length mismatch"

        # 5) 固定本轮 n 与维度（若尚未固定）
        st_changed = False
        if not st["n_hex"]:
            st["n_hex"] = tx.paillier_n_hex
            n = int(st["n_hex"], 16);
            st["n2_hex"] = hex(n * n)[2:]
            if st["w_len"] == 0: st["w_len"] = tx.w_len
            if st["b_len"] == 0: st["b_len"] = tx.b_len
            if not st["agg_paillier_cw_hex"] or len(st["agg_paillier_cw_hex"]) != tx.w_len:
                st["agg_paillier_cw_hex"] = ["1"] * tx.w_len
            st_changed = True
        else:
            if st["n_hex"] != tx.paillier_n_hex:
                self._round_stat(tx.round, accepted=False)
                return False, "paillier modulus mismatch"
            if st["w_len"] != tx.w_len or st["b_len"] != tx.b_len:
                self._round_stat(tx.round, accepted=False)
                return False, "dimension mismatch with round state"

        # 6) 签名校验（secp256k1）
        try:
            vk = VerifyingKey.from_string(unhexlify(tx.pubkey_hex), curve=SECP256k1)
            vk.verify(unhexlify(tx.sig_hex), tx.hash_for_sign())
        except (BadSignatureError, Exception):
            self._round_stat(tx.round, accepted=False)
            return False, "bad signature"

        # 7) Paillier 密文聚合：逐元素 c_agg[i] = c_agg[i] * c_i (mod n^2)
        n2 = int(st["n2_hex"] or "0", 16)
        if n2 == 0:
            self._round_stat(tx.round, accepted=False)
            return False, "round n not set"
        new_agg = []
        for i, c_hex in enumerate(tx.paillier_cw_hex):
            c_prev = int(st["agg_paillier_cw_hex"][i], 16)
            c_i = int(c_hex, 16)
            # 乘法聚合（同态加法），对应论文式(28):contentReference[oaicite:8]{index=8}
            c_new = (c_prev * c_i) % n2
            new_agg.append(hex(c_new)[2:])
        st["agg_paillier_cw_hex"] = new_agg

        # 8) CKKS“加法序列”占位：把偏置密文载荷追加到日志（链下 CKKS 执行器按顺序重放 add）
        st["ckks_add_log"].append(tx.ckks_cb_payload)

        # 9) 标记 nonce、参与者，记账
        st["used_nonces"][tx.nonce] = True
        st["participants"].append(tx.client_id)
        st["accepted"] += 1

        # 10) 保存 TX 与轮次状态
        txid = self._txid(tx)
        self.kv.put(self._key_tx(txid), json.dumps(asdict(tx)))
        self.kv.put(self._key_round(tx.round), json.dumps(st))
        if st_changed:
            # 保证变更已提交
            pass

        return True, txid

    # ---- 结束轮次并导出聚合结果 ----
    def finalize_round(self, r: int) -> Dict:
        st = self._load_round(r)
        if not st or st["close_ts"] is not None:
            return {"ok": False, "err": "round not open or not found"}
        st["close_ts"] = int(time.time())
        self.kv.put(self._key_round(r), json.dumps(st))

        # 切到下一轮（n/w/b 由下一轮第一笔交易固定）
        self.open_round(r + 1, n_hex="", w_len=0, b_len=0)
        self.kv.put(self._key_head(), str(r + 1))

        # 返回本轮“加法聚合产物”与 CKKS 序列（供离线解密/重放以及审计）
        return {
            "ok": True,
            "round": r,
            "agg_paillier_cw_hex": st["agg_paillier_cw_hex"],
            "paillier_n_hex": st["n_hex"],
            "ckks_add_log": st["ckks_add_log"],
            "participants": st["participants"],
            "accepted": st["accepted"],
            "rejected": st["rejected"],
            "close_ts": st["close_ts"]
        }

    # ---- 查询 ----
    def get_round(self, r: int) -> Optional[Dict]:
        return self._load_round(r)

    def get_tx(self, txid: str) -> Optional[Dict]:
        v = self.kv.get(self._key_tx(txid))
        return json.loads(v) if v else None

    def verify_tx(self, txid: str) -> bool:
        tx = self.get_tx(txid)
        if not tx: return False
        utx = UpdateTx(**tx)
        try:
            vk = VerifyingKey.from_string(unhexlify(utx.pubkey_hex), curve=SECP256k1)
            vk.verify(unhexlify(utx.sig_hex), utx.hash_for_sign())
            return True
        except Exception:
            return False

    # ---- 内部工具 ----
    def _txid(self, tx: UpdateTx) -> str:
        h = hashlib.sha256(tx.hash_for_sign() + unhexlify(tx.sig_hex)).hexdigest()
        return f"0x{h}"

    def _load_round(self, r: int) -> Optional[Dict]:
        v = self.kv.get(self._key_round(r))
        return json.loads(v) if v else None

    def _round_stat(self, r: int, accepted: bool):
        st = self._load_round(r)
        if not st: return
        st["accepted"] += 1 if accepted else 0
        st["rejected"] += 0 if accepted else 1
        self.kv.put(self._key_round(r), json.dumps(st))



if __name__ == "__main__":
    kv = KVStore()
    sc = HHBFLContract(kv)


    print("Head round:", sc.head_round())
    # 单元测试建议见 README.md
