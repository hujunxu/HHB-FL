import fiona
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def load_data_fiona(gdb_path, layer_name):
    """
    使用Fiona库加载Geodatabase文件并转换为GeoDataFrame。

    参数:
    gdb_path (str): Geodatabase文件的路径。
    layer_name (str): 要加载的层名称。

    返回:
    gpd.GeoDataFrame: 加载的GeoDataFrame。
    """
    start_time = time.time()
    try:
        logging.info("开始加载数据...")
        with fiona.open(gdb_path, layer=layer_name) as src:
            data = []
            for feature in tqdm(src, total=len(src)):
                data.append(feature)
            gdf = gpd.GeoDataFrame.from_features(data)
            logging.info(f"数据加载完成，用时 {time.time() - start_time:.2f} 秒")
            return gdf
    except Exception as e:
        logging.error(f"加载数据时出错: {e}")
        return None


def preprocess_data(gdf):
    """
    清洗和标准化数据。

    参数:
    gdf (gpd.GeoDataFrame): 输入的GeoDataFrame。

    返回:
    gpd.GeoDataFrame: 预处理后的GeoDataFrame。
    """
    start_time = time.time()
    logging.info("开始预处理数据...")

    # 打印几何类型
    geom_types = gdf.geometry.type
    logging.info(f"几何类型: {geom_types.unique()}")

    # 检查并处理MultiLineString类型
    if all(geom_types == 'MultiLineString'):
        gdf['geometry'] = gdf.geometry.centroid
        logging.info("已将MultiLineString几何转换为质心（Point）")

    # 提取经纬度信息
    gdf['longitude'] = gdf.geometry.x
    gdf['latitude'] = gdf.geometry.y

    # 打印数据集中的列名
    logging.info(f"数据集中的列名: {gdf.columns.tolist()}")

    required_columns = ['longitude', 'latitude', 'Length', 'Width', 'Draft']
    for col in required_columns:
        if col not in gdf.columns:
            raise ValueError(f"数据中未找到列 '{col}'。")

    # 仅填充数值列中的缺失值
    num_cols = gdf.select_dtypes(include=[np.number]).columns
    gdf[num_cols] = gdf[num_cols].fillna(gdf[num_cols].mean())

    for col in required_columns:
        gdf[col] = (gdf[col] - gdf[col].mean()) / gdf[col].std()

    logging.info(f"数据预处理完成，用时 {time.time() - start_time:.2f} 秒")
    return gdf


def save_data(gdf, output_path):
    """
    将预处理后的GeoDataFrame保存为CSV文件。

    参数:
    gdf (gpd.GeoDataFrame): 预处理后的GeoDataFrame。
    output_path (str): 保存CSV文件的路径。
    """
    start_time = time.time()
    try:
        logging.info("开始保存数据...")
        gdf.to_csv(output_path, index=False)
        logging.info(f"数据保存完成，用时 {time.time() - start_time:.2f} 秒")
    except Exception as e:
        logging.error(f"保存数据时出错: {e}")


if __name__ == "__main__":
    gdb_path = r"E:\桌面\实验\GreatLakes.gdb"  # 替换为你的Geodatabase文件路径
    layer_name = "Tracks_2015_01"  # 替换为你的实际层名称
    output_path = "preprocessed_data.csv"

    start_time = time.time()
    logging.info("开始整个数据处理流程...")
    gdf = load_data_fiona(gdb_path, layer_name)
    if gdf is not None:
        gdf = preprocess_data(gdf)
        save_data(gdf, output_path)
        logging.info(f"整个数据处理流程完成，用时 {time.time() - start_time:.2f} 秒")
