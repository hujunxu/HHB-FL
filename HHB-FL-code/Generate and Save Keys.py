import os
import pickle
import tenseal as ts
from phe import paillier

# Define directories
data_dir = r'E:\桌面\实验\Client_data'

# Generate and save Paillier keys
public_key, private_key = paillier.generate_paillier_keypair()

with open(os.path.join(data_dir, 'paillier_private_key.pkl'), 'wb') as key_file:
    pickle.dump(private_key, key_file)
with open(os.path.join(data_dir, 'paillier_public_key.pkl'), 'wb') as key_file:
    pickle.dump(public_key, key_file)

# Generate and save CKKS context
ckks_context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
ckks_context.generate_galois_keys()
ckks_context.generate_relin_keys()

with open(os.path.join(data_dir, 'ckks_context.pkl'), 'wb') as context_file:
    context_file.write(ckks_context.serialize())
