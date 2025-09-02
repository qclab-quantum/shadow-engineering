from collections import defaultdict
from jax import vmap
import jax.numpy as jnp 
import numpy as np
import pennylane as qml
from pennylane import numpy as np
from pennylane.shadows import ClassicalShadow
from functools import partial
import jax
import time,itertools,math
from functools import partial
from jax.experimental import sparse
import pennylane as qml
import jax.numpy as jnp
from functools import reduce
import numpy as np
from jax.experimental.sparse import BCOO
# from skrmt.ensemble import CircularEnsemble
import math,itertools
from scipy.sparse import csr_matrix
import pennylane as qml
from scipy.linalg import expm
import time
from joblib import Parallel, delayed, parallel_backend
from scipy.sparse import coo_matrix

from jax import jit
Pauli_list = {
    6: jnp.array([[1, 0], [0, 1]]),
    7 : jnp.array([[0, 1], [1, 0]]),
    8 : jnp.array([[0, -1j], [1j, 0]]),
    9 : jnp.array([[1, 0], [0, -1]])
}
stab1 = {
    0: jnp.array([[1, 0], [0, 0]]),  # |0⟩⟨0|
    1: jnp.array([[0, 0], [0, 1]]),  # |1⟩⟨1|
    2: 0.5 * jnp.array([[1, 1], [1, 1]]),  # |+⟩⟨+|
    3: 0.5 * jnp.array([[1, -1], [-1, 1]]),  # |-⟩⟨-|
    4: 0.5 * jnp.array([[1, -1j], [1j, 1]]),  # |i⟩⟨i|
    5: 0.5 * jnp.array([[1, 1j], [-1j, 1]]), # |-i⟩⟨-i|
}
stab2 = {
    0: jnp.array([[2, 0], [0, -1]]),  # 3|0⟩⟨0|-I
    1: jnp.array([[-1, 0], [0, 2]]),  # 3|1⟩⟨1|-I
    2: 0.5 * jnp.array([[1, 3], [3, 1]]),  # 3|+⟩⟨+|-I
    3: 0.5 * jnp.array([[1, -3], [-3, 1]]),  # 3|-⟩⟨-|-I
    4: 0.5 * jnp.array([[1, -3j], [3j, 1]]),  # 3|i⟩⟨i|-I
    5: 0.5 * jnp.array([[1, 3j], [-3j, 1]]), # 3|-i⟩⟨-i|-I
}
state_vectors = {
    0: np.array([1, 0]),
    1: np.array([0, 1]),
    2: np.array([1, 1]) / np.sqrt(2),
    3: np.array([1, -1]) / np.sqrt(2),
    4: np.array([1, 1j]) / np.sqrt(2),
    5: np.array([1, -1j]) / np.sqrt(2),
}


'''Translate the input_list [0-5 (as strings) and 6-9 (as Pauli operators)] into density matrix.'''
def str2rho(input_list):
    @jax.jit
    def get_matrices(input_list):
        def map_index_to_matrix(index):
            cases = [lambda: jnp.array([[1, 0], [0, 0]], dtype=jnp.complex64),
                    lambda: jnp.array([[0, 0], [0, 1]], dtype=jnp.complex64),
                    lambda: 0.5 * jnp.array([[1, 1], [1, 1]], dtype=jnp.complex64),
                    lambda: 0.5 * jnp.array([[1, -1], [-1, 1]], dtype=jnp.complex64),
                    lambda: 0.5 * jnp.array([[1, -1j], [1j, 1]], dtype=jnp.complex64),
                    lambda: 0.5 * jnp.array([[1, 1j], [-1j, 1]], dtype=jnp.complex64),
                    lambda: jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64),
                    lambda: jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64),
                    lambda: jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64),
                    lambda: jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
                    ]
            return jax.lax.switch(index, cases)
        return jax.lax.map(map_index_to_matrix, jnp.array(input_list))
    @jax.jit
    def tensor_product(matrices):
        return reduce(jnp.kron, matrices)
    result = tensor_product(get_matrices(input_list))
    return result

'''Obtain the density matrix on stab2 from input_list (0-5).'''
def str2rho_stab2(input_list):
    @jax.jit
    def get_matrices(input_list):
        def map_index_to_matrix(index):
            cases = [lambda: jnp.array([[2, 0], [0, -1]], dtype=jnp.complex64),
                    lambda: jnp.array([[-1, 0], [0, 2]], dtype=jnp.complex64),
                    lambda: 0.5 * jnp.array([[1, 3], [3, 1]], dtype=jnp.complex64),
                    lambda: 0.5 * jnp.array([[1, -3], [-3, 1]], dtype=jnp.complex64),
                    lambda: 0.5 * jnp.array([[1, -3j], [3j, 1]], dtype=jnp.complex64),
                    lambda: 0.5 * jnp.array([[1, 3j], [-3j, 1]], dtype=jnp.complex64)
                    ]
            return jax.lax.switch(index, cases)
        return jax.lax.map(map_index_to_matrix, jnp.array(input_list))
    @jax.jit
    def tensor_product(matrices):
        return reduce(jnp.kron, matrices)
    result = tensor_product(get_matrices(input_list))
    return result

'''Translate the input 0-5 (as strings) into quantum states.'''
def str2rhoinistr(input_list):
    @jax.jit
    def get_matrices(input_list):
        def map_index_to_matrix(index):
            cases = [lambda: jnp.array([1, 0], dtype=jnp.complex64),
                    lambda: jnp.array( [0, 1], dtype=jnp.complex64),
                    lambda: jnp.array([1, 1]/ np.sqrt(2), dtype=jnp.complex64),
                    lambda: jnp.array([1, -1]/ np.sqrt(2), dtype=jnp.complex64),
                    lambda: jnp.array([1, 1j]/ np.sqrt(2), dtype=jnp.complex64),
                    lambda: jnp.array([1, -1j]/ np.sqrt(2), dtype=jnp.complex64),

                    ]
            return jax.lax.switch(index, cases)
        return jax.lax.map(map_index_to_matrix, jnp.array(input_list))
    @jax.jit
    def tensor_product(matrices):
        return reduce(jnp.kron, matrices)
    result = tensor_product(get_matrices(input_list))
    return jnp.array(result)

'''Compute the element-wise tensor product of the density matrix list.'''
def kron_product_chain(matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = jnp.kron(result, matrix)
    return result

'''Perform 3X-I processing on each element of the density matrix list, then compute their tensor products one by one.'''
def kron_str(strrho):
    kron_result = 3*stab1[strrho[0]]-jnp.identity(2)
    for char in strrho[1:]:
        kron_result = jnp.kron(kron_result, 3*stab1[char]-jnp.identity(2))
    return kron_result
def tensor_product(*matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = jnp.kron(result, matrix)
    return result

'''Count the number of non-I elements in the list.'''
@jax.jit
def count_non_I(pauli_string):
    non_I_mask = jnp.not_equal(pauli_string, 6)
    non_I_count = jnp.sum(non_I_mask)
    return non_I_count
'''For the matrix equation AX = B, calculate X.'''
@jax.jit
def calculate_T(A, B):
    A_flat = vmap(jnp.ravel)(A)
    B_flat = vmap(jnp.ravel)(B)
    B_inv = jnp.linalg.inv(B_flat.T)
    a_matrix = jnp.dot(A_flat, B_inv)
    return a_matrix



'''Calculate the function C(k, d)'''
def cal_C_k_d(k, d):
    def factorial(n):
        if n == 0: return 1  
        else: return n * factorial(n-1)
    C_k_d = math.sqrt(2 * factorial(k)) / (math.sqrt(d) * k**(k+2.5) * (2*math.sqrt(6) + 4*math.sqrt(3))**k)  
    return C_k_d


'''Calculate the filtered coefficient alpha_p based on the filtering parameter tilde_epsilon.'''
@jax.jit
def calculate_alpha_p(x_p, beta_p, tilde_epsilon, eta):
    cond1 = beta_p <= 2 * tilde_epsilon
    cond2 = jnp.logical_and(beta_p > 2 * tilde_epsilon, jnp.abs(x_p) / jnp.sqrt(beta_p) <= 2 * eta * jnp.sqrt(tilde_epsilon))
    result = jax.lax.cond(cond1, lambda _: jnp.asarray(0., dtype=jnp.complex64), lambda _: jax.lax.cond(cond2, lambda _: jnp.asarray(0., dtype=jnp.complex64), lambda _: (x_p / beta_p).astype(jnp.complex64), None), None)
    return result


'''Calculate the mean squared error.'''
@jax.jit
def Rmean_squared_error(y_true, y_pred):
    mse = jnp.mean((y_pred - y_true) ** 2).real
    return jnp.sqrt(jnp.maximum(mse, 0))

'''Obtain the Hamiltonians of one-dimensional (1D) n-spin XY and Ising chains with homogeneous or disordered Z fields.'''
def Hamiltonian(h,model,num_qubits):
    H = jnp.zeros((2**num_qubits, 2**num_qubits), dtype=complex)  # 初始化哈密顿量矩阵
    for i in range(num_qubits-1):
        XiXi1 = tensor_product(*[Pauli_list[7] if j == i or j == i+1 else jnp.identity(2) for j in range(num_qubits)])
        if model == "XY":
            YiYi1 = tensor_product(*[Pauli_list[8] if j == i or j == i+1 else jnp.identity(2) for j in range(num_qubits)])
            H += 0.25 * (XiXi1 + YiYi1)
        if model == "Ising":
            YiYi1 = 0
        H += 0.25 * (XiXi1 + YiYi1) 
    for i in range(num_qubits):
        Z_i = tensor_product(*[Pauli_list[9] if j == i else jnp.identity(2) for j in range(num_qubits)])
        H += 0.5 * Z_i * h[i]
    return H




def XY_model_hamiltonian(h, num_qubits):
    coeffs = []  # 系数列表
    obs = []     # 可观测量列表

    # 添加 X_i X_{i+1} 和 Y_i Y_{i+1} 项
    for i in range(num_qubits - 1):
        # X_i X_{i+1} 项
        XiXi1 = qml.PauliX(i) @ qml.PauliX(i + 1)
        # Y_i Y_{i+1} 项
        YiYi1 = qml.PauliY(i) @ qml.PauliY(i + 1)
        # 添加系数和可观测量
        coeffs.extend([0.25, 0.25])
        obs.extend([XiXi1, YiYi1])

    # 添加 h_i Z_i 项
    for i in range(num_qubits):
        Z_i = qml.PauliZ(i)
        coeffs.append(0.5 * h[i])
        obs.append(Z_i)

    # 创建 PennyLane 哈密顿量
    return qml.Hamiltonian(coeffs, obs)


'''The observable H in PennyLane can be converted into a matrix via qml.matrix.'''
def calculate_observable(n, J_x, J_y, J_z, h, c):
    coeffs = []
    terms = []

    # 邻近比特之间的两体相互作用
    for j in range(n - 1):
        # XX 项
        xx_term = [qml.PauliX(k) if k in [j, j + 1] else qml.Identity(k) for k in range(n)]
        xx_term = qml.prod(*xx_term)
        coeffs.append(J_x)
        terms.append(xx_term)
        yy_term = [qml.PauliY(k) if k in [j, j + 1] else qml.Identity(k) for k in range(n)]
        yy_term = qml.prod(*yy_term)
        coeffs.append(J_y)
        terms.append(yy_term)
        zz_term = [qml.PauliZ(k) if k in [j, j + 1] else qml.Identity(k) for k in range(n)]
        zz_term = qml.prod(*zz_term)
        coeffs.append(J_z)
        terms.append(zz_term)

    for j in range(n):
        z_term = [qml.PauliZ(k) if k == j else qml.Identity(k) for k in range(n)]
        z_term = qml.prod(*z_term)
        coeffs.append(h)
        terms.append(z_term)

    # 常数项
    identity_term = [qml.Identity(k) for k in range(n)]
    identity_term = qml.prod(*identity_term)
    coeffs.append(c)
    terms.append(identity_term)

    # obs_hamiltonian = qml.Hamiltonian(coeffs, terms)
    return coeffs, terms

def convert_terms(terms, n):
    converted = []
    for term in terms:
        current = [6] * n
        for factor in term.operands:
            wires = factor.wires.tolist()
            if not wires:
                continue
            pos = wires[0]
            if isinstance(factor, qml.PauliX):
                current[pos] = 7
            elif isinstance(factor, qml.PauliY):
                current[pos] = 8
            elif isinstance(factor, qml.PauliZ):
                current[pos] = 9
        converted.append(current)
    return converted
SINGLE_TRACES_ARRAY_stab2 = jnp.array([
    [1,  1,  1,  1,  1,  1],
    [0,  0,  3, -3,  0,  0],
    [0,  0,  0,  0,  3, -3],
    [3, -3,  0,  0,  0,  0]
], dtype=jnp.float32)  # 确保数据类型一致
def optimized_compute_multi_trace_stab2(all_pauli, rho_input):
    # 转换Pauli索引到0-3范围

    pauli_indices = all_pauli - 6  # 直接向量化操作
    trace_values = SINGLE_TRACES_ARRAY_stab2[pauli_indices, rho_input]

    prod_per_row = jnp.prod(trace_values, axis=1)
    has_zero = jnp.any(trace_values == 0, axis=1)
    sita = jnp.where(has_zero, 0.0, prod_per_row)

    return sita


def tensor_network_find_keys(shadow):
    mapping_keys = jnp.array([2, 4, 0, 3, 5, 1], dtype=jnp.int32)
    mapping_values = jnp.array([
        [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]
    ], dtype=jnp.int32)
    
    first_part = shadow[0][0].flatten().astype(jnp.int32)
    second_part = shadow[1][0].flatten().astype(jnp.int32)
    
    features = jnp.stack([first_part, second_part], axis=1)
    matches = jnp.all(features[:, None, :] == mapping_values[None, :, :], axis=2)
    key_indices = jnp.argmax(matches.astype(jnp.int32), axis=1)
    return mapping_keys[key_indices]

@jit
def shadow2index(shadow):
    return tensor_network_find_keys(shadow)
@jit
def shadow2index2(shadowbits, shadowrecipes):
    mapping_keys = jnp.array([2, 4, 0, 3, 5, 1], dtype=jnp.int32)
    mapping_values = jnp.array([
        [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]
    ], dtype=jnp.int32)
    
    first_part = shadowbits.flatten().astype(jnp.int32)
    second_part = shadowrecipes.flatten().astype(jnp.int32)
    
    features = jnp.stack([first_part, second_part], axis=1)
    matches = jnp.all(features[:, None, :] == mapping_values[None, :, :], axis=2)
    key_indices = jnp.argmax(matches.astype(jnp.int32), axis=1)
    return mapping_keys[key_indices]
@jax.jit
def list_to_base6_jax(keys):
    # Convert the list of digits to a base-6 number
    base6_number = jnp.sum(keys * (6 ** jnp.arange(len(keys))[::-1]))
    return base6_number


'''This converts the indices on stab1 to their corresponding shadows, in order to apply the following method of calculating expected values using shadows.'''
@partial(jax.jit, static_argnums=(1,))
def index_to_keys(index: int, n: int = 3) -> jnp.ndarray:
    exponents = n - 1 - jnp.arange(n)
    divisors = 6 ** exponents
    digits = (index // divisors) % 6
    return digits.astype(jnp.int32)
@jax.jit
def keys_to_shadow(keys: jnp.ndarray) -> tuple:
    mapping_keys = jnp.array([2, 4, 0, 3, 5, 1])
    mapping_values = jnp.array([
        [0, 0],# 加态
        [0, 1],#y+
        [0, 2],#0
        [1, 0],#-

        [1, 1],#y-
        [1, 2]#1
    ], dtype=jnp.int8)
    def get_value(key):
        index = jnp.argmax(mapping_keys == key)
        return mapping_values[index]
    values = jax.vmap(get_value)(keys)
    return [[values[:, 0].astype(jnp.int8)], [values[:, 1].astype(jnp.int8)]]

@partial(jax.jit, static_argnums=(1,))
def index_to_shadow(index: int, num_qubits) -> tuple:
    keys = index_to_keys(index, num_qubits)
    return keys_to_shadow(keys)


'''Construct the full Z observable.'''
def calculate_observable_zglobal(n):
    coeffs = [1.0] 
    terms = [qml.prod(*[qml.PauliZ(i) for i in range(n)])]
    return qml.Hamiltonian(coeffs, terms)


'''Calculate the expected value of the product of Pauli operators and six states.
pauli_indices is a list of length num_qubits consisting of [6,7,8,9]
rho_indices is a list of length num_qubits consisting of [0,1,2,3,4,5]'''
SINGLE_TRACES_ARRAY = jnp.array([
    [1,  1,  1,  1,  1,  1],
    [0,  0,  1, -1,  0,  0],
    [0,  0,  0,  0,  1, -1],
    [1, -1,  0,  0,  0,  0]
])

@jax.jit
def compute_multi_trace(pauli_indices, rho_indices):
    trace_values = SINGLE_TRACES_ARRAY[pauli_indices - 6, rho_indices]
    has_zero = jnp.any(trace_values == 0)
    return jnp.where(has_zero, 0.0, jnp.prod(trace_values))


@jax.jit
def calculate_alpha_p_vec(x_p, beta_p, tilde_epsilon, eta):
    cond1 = beta_p <= 2 * tilde_epsilon
    abs_xp_over_sqrt_beta = jnp.abs(x_p) / jnp.sqrt(beta_p)
    cond2 = (beta_p > 2 * tilde_epsilon) & (abs_xp_over_sqrt_beta <= 2 * eta * jnp.sqrt(tilde_epsilon))
    return jnp.where(cond1 | cond2, 0.0, x_p / beta_p).astype(jnp.complex64)

def process_block(pauli_indices, rhoin_block, rhoout_block):
    trace_values = SINGLE_TRACES_ARRAY[pauli_indices[:, None, :], rhoin_block[None, :, :]]  # (num_paulis, block_size, n)
    has_zero = jnp.any(trace_values == 0, axis=-1)  # (num_paulis, block_size)
    product = jnp.prod(trace_values, axis=-1)       # (num_paulis, block_size)
    prhoin_p = jnp.where(has_zero, 0.0, product)    # (num_paulis, block_size)
    xp_block = jnp.einsum('pm,m->p', prhoin_p, rhoout_block) / rhoout_block.shape[0]  # (num_paulis,)
    return xp_block


'''Batch processing Pauli operators and density matrices (rho) of equal length'''
def optimized_compute_multi_trace(all_pauli, rho_input):
    pauli_indices = all_pauli - 6
    trace_values = SINGLE_TRACES_ARRAY[pauli_indices, rho_input]
    prod_per_row = jnp.prod(trace_values, axis=1)
    has_zero = jnp.any(trace_values == 0, axis=1)
    sita = jnp.where(has_zero, 0.0, prod_per_row)

    return sita



'''The main function is calculate_shadow_expvals. 
Given the shadow containing [bits, recipes] = [[values[:, 0].astype(jnp.int8)], [values[:, 1].astype(jnp.int8)]] (where bits takes values 0 and 1, and recipes takes values 0, 1, 2 representing X, Y, Z respectively) and H as the observable (a PennyLane observable list), this function calculates the expected value of H.'''
def calculate_shadow_expvals(shadows, H):
    bits = [shadow.bits for shadow in shadows] 
    recipes = [shadow.recipes for shadow in shadows]
    bits = np.array(bits)
    bits = jnp.array(bits)
    recipes = np.array(recipes)
    recipes = jnp.array(recipes)
    nqubits = bits.shape[2]
    coeffs_and_words = shadows[0]._convert_to_pauli_words(H)
    words_jnp = jnp.array([coeff_and_word[1] for coeff_and_word in coeffs_and_words])
    words_jnp = jnp.reshape(words_jnp, (len(H),1,nqubits))
    coeffs_jnp = jnp.array([coeff_and_word[0] for coeff_and_word in coeffs_and_words])
    e = compute_expvals_for_obs(bits, recipes, words_jnp)
    coeffs_jnp_reshaped = coeffs_jnp.reshape(len(H),1)
    return jnp.concatenate(e.T @ coeffs_jnp_reshaped, axis=0)

'''Calculate the expected value of the output state under the observable based on the measured bit string (01) and the selected Pauli basis (XYZ).'''
def compute_expvals_for_obs(bits, recipes, words):
    def compute_expvals(bits, recipes, word):
        T, n = recipes.shape
        b = word.shape[0]
        id_mask = word == -1
        indices = jnp.equal(
            jnp.reshape(recipes, (T, 1, n)), jnp.reshape(word, (1, b, n))
        )
        indices = jnp.logical_or(indices, jnp.tile(jnp.reshape(id_mask, (1, b, n)), (T, 1, 1)))
        indices = jnp.all(indices, axis=2)
        bits = jnp.where(id_mask, 0, jnp.tile(jnp.expand_dims(bits, 1), (1, b, 1)))
        bits = jnp.sum(bits, axis=2) % 2
        expvals = jnp.mean(jnp.where(indices, 1 - 2 * bits, 0) * 3 ** jnp.count_nonzero(
            jnp.logical_not(id_mask), axis=1
        ))
        return jnp.float32(expvals)
    def compute_expvals_with_single_ob(bits, recipes, word):
        return vmap(compute_expvals, in_axes=(0,0,None))(bits, recipes, word)
    return vmap(compute_expvals_with_single_ob, in_axes=(None,None,0))(bits, recipes, words)

def compute_expvals_with_single_ob(bits, recipes, word):
    def compute_expvals(bits, recipes, word):
        T, n = recipes.shape
        b = word.shape[0]
        id_mask = word == -1
        indices = jnp.equal(
            jnp.reshape(recipes, (T, 1, n)), jnp.reshape(word, (1, b, n))
        )
        indices = jnp.logical_or(indices, jnp.tile(jnp.reshape(id_mask, (1, b, n)), (T, 1, 1)))
        indices = jnp.all(indices, axis=2)
        bits = jnp.where(id_mask, 0, jnp.tile(jnp.expand_dims(bits, 1), (1, b, 1)))
        bits = jnp.sum(bits, axis=2) % 2
        expvals = jnp.mean(jnp.where(indices, 1 - 2 * bits, 0) * 3 ** jnp.count_nonzero(
            jnp.logical_not(id_mask), axis=1
        ))
        return jnp.float32(expvals)
    return vmap(compute_expvals, in_axes=(0,0,None))(bits, recipes, word)

'''Generate a Haar random quantum state.'''
@partial(jax.jit, static_argnames=('n_qubits',))
def generate_haar_random_state(n_qubits, key):
    dim = 2 ** n_qubits
    key_real, key_imag = jax.random.split(key)
    X = jax.random.normal(key_real, (dim, dim)) + 1j * jax.random.normal(key_imag, (dim, dim))
    Q, R = jnp.linalg.qr(X)
    phases = jnp.diag(R) / jnp.abs(jnp.diag(R))
    Q = Q * jnp.conj(phases)[:, jnp.newaxis]
    state = Q[:, 0]
    return state / jnp.linalg.norm(state)
@partial(jax.jit, static_argnames=('n_qubits',))
def haardensitymatrix(n_qubits):
    key = jax.random.PRNGKey(int(time.time())) 
    state = generate_haar_random_state(n_qubits, key)
    return jnp.outer(state, jnp.conj(state))

'''Remove the non-zero rows and non-zero columns from the scipy_coo sparse matrix.'''
def remove_zero_rows_cols_coo(matrix):
    coo = matrix.tocoo()
    non_zero_rows = np.unique(coo.row)
    non_zero_cols = np.unique(coo.col)
    filtered_matrix = matrix.tocsr()[non_zero_rows][:, non_zero_cols]
    return filtered_matrix

'''Only retain the specified zero rows and specified zero columns in the scipy_coo sparse matrix.'''
def keep_specific_rows_cols_coo(scipy_coo, rows_to_keep, cols_to_keep):

    data = jnp.array(scipy_coo.data)
    rows = jnp.array(scipy_coo.row, dtype=jnp.int32)
    cols = jnp.array(scipy_coo.col, dtype=jnp.int32)

    rows_to_keep = jnp.unique(jnp.array(rows_to_keep, dtype=jnp.int32))
    cols_to_keep = jnp.unique(jnp.array(cols_to_keep, dtype=jnp.int32))
    row_mask = jnp.isin(rows, rows_to_keep)
    col_mask = jnp.isin(cols, cols_to_keep)
    mask = row_mask & col_mask
    filtered_data = data[mask]
    filtered_rows = rows[mask]
    filtered_cols = cols[mask]

    sorted_rows = jnp.sort(rows_to_keep)
    sorted_cols = jnp.sort(cols_to_keep)
    
    new_rows = jnp.searchsorted(sorted_rows, filtered_rows).astype(jnp.int32)
    new_cols = jnp.searchsorted(sorted_cols, filtered_cols).astype(jnp.int32)

    return BCOO(
        (filtered_data, jnp.column_stack([new_rows, new_cols])),
        shape=(len(rows_to_keep), len(cols_to_keep))
    )

'''Calculate the ideal transition matrix of the channel.'''
@partial(jax.jit,static_argnums=(0,))
def get_prob(num_qubits,Umatrix,rhoin6):
    def compute_rho_true(kron):
        return Umatrix @ kron @ Umatrix.conj().T  
    rho_true_all = jax.vmap(compute_rho_true)(rhoin6)
    def compute_probs(k):
        rho_true = rho_true_all[k]
        def inner_loop(i):
            return ((1/3)**num_qubits) * jnp.trace(rho_true @ rhoin6[i]) 
        return jax.vmap(inner_loop)(jnp.arange(6**num_qubits))
    probs = jax.vmap(compute_probs)(jnp.arange(6**num_qubits))
    return probs
'''Calculate the output states.'''
@jax.jit
def calrho3(U_matrix, rhoinput):
    return U_matrix@rhoinput@U_matrix.conj().T

'''Generate a list of Pauli strings where the local value is less than or equal to k.'''
def generate_pauli(num_qubits, k):
    result = []
    max_t = min(k, num_qubits)
    for t in range(max_t + 1):
        if t == 0:
            result.append(jnp.full((1, num_qubits), 6, dtype=jnp.int32))
            continue
        pos_combs = np.array(list(itertools.combinations(range(num_qubits), t)))
        C = pos_combs.shape[0]
        val_combs = jnp.array(list(itertools.product([7, 8, 9], repeat=t)))
        M = val_combs.shape[0] 
        base = jnp.full((C * M, num_qubits), 6, dtype=jnp.int32)
        pos_expanded = jnp.repeat(pos_combs, M, axis=0)  
        val_expanded = jnp.tile(val_combs, (C, 1))  
        row_indices = jnp.arange(C * M)[:, None]  
        col_indices = pos_expanded  
        values = val_expanded  
        base = base.at[row_indices, col_indices].set(values)
        result.append(base)
    return jnp.concatenate(result, axis=0) if result else jnp.empty((0, num_qubits), dtype=jnp.int32)




import random
def generate_random_integers(nshot, num_qubits):
    upper_bound = 6 ** num_qubits
    random_integers = [random.randint(0, upper_bound - 1) for _ in range(nshot)]
    return random_integers

'''Normalize the sparse transfer matrix'''
def normalize_coo_matrix(coo_mat):
    row, col, data = coo_mat.row, coo_mat.col, coo_mat.data
    row_sums = np.zeros(coo_mat.shape[0])
    np.add.at(row_sums, row, data)
    normalized_data = data / row_sums[row]
    return coo_matrix((normalized_data, (row, col)), shape=coo_mat.shape)

def generate_random_sublists(nshot, num_qubits, key= jax.random.PRNGKey(int(time.time()))):
    return jax.random.randint(key, (nshot, num_qubits), 0, 6)



'''Construct a COO matrix based on resultin and resultout.'''
def build_coo_matrix(resultin, resultout):
    # 步骤1：收集唯一输入/输出态并建立映射
    unique_ins = {tuple(state): idx for idx, state in enumerate(sorted({tuple(s) for s in resultin}))}
    unique_outs = {tuple(state): idx for idx, state in enumerate(sorted({tuple(s) for s in resultout}))}
    rows, cols, data = [], [], []
    counter = defaultdict(int)
    for in_state, out_state in zip(resultin, resultout):
        counter[(tuple(in_state), tuple(out_state))] += 1
    for (in_t, out_t), count in counter.items():
        rows.append(unique_ins[in_t])
        cols.append(unique_outs[out_t])
        data.append(count)
    coo_mtx = coo_matrix(
        (data, (rows, cols)),
        shape=(len(unique_ins), len(unique_outs)),
        dtype=np.int32
    )
    return coo_mtx

'''The dictionary representation of the transformation matrix from stab1 to stab2.'''
def construct_base_matrix_dict():
    return [
         {0: 2, 1: -1},
       {0: -1, 1: 2},
        {0: -1, 1: -1, 2: 3},
        {0: -1, 1: -1, 3: 3},
        {0: -1, 1: -1, 4: 3},
        {0: -1, 1: -1, 5: 3},
    ]

'''Construct a sparse matrix that can be switched to CSR storage.'''
def construct_sparse_matrix(resultin, resultout,n_bits):
    M_dict = np.array(construct_base_matrix_dict())
    resultin = np.asarray(resultin, dtype=int)
    resultout = np.asarray(resultout, dtype=int)
    resultin,_ = np.unique(resultin, axis=0, return_inverse=True)
    resultout, _ = np.unique(resultout, axis=0, return_inverse=True)
    nshot, nout = len(resultin), len(resultout)
    target_cols = resultout 

    rows, cols, data = [], [], []
    for i in range(nshot):
        row_indices = resultin[i] 
        for j in range(nout):
            target_digits = target_cols[j] 
            valid = True
            product = 1
            for k in range(n_bits):
                row_k = row_indices[k]     
                digit_k = target_digits[k]
                if digit_k not in M_dict[row_k]:
                    valid = False
                    break
                product *= M_dict[row_k][digit_k]
            if valid:
                rows.append(i)
                cols.append(j)
                data.append(product)
    return coo_matrix((data, (rows, cols)), shape=(nshot, nout), dtype=np.int8)

'''Generate all possible combinations of elements on stab1'''
def vakey(num_qubits):
    combinations = itertools.product([0, 1, 2, 3, 4, 5], repeat=num_qubits)
    keys_list = [list(combination) for combination in combinations]
    return jnp.array(keys_list)

'''Load the resultin and resultout data from the filename.'''
def load_results(num_qubits, filename=None):
    data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=int)
    exp_ids = np.unique(data[:, 0])
    all_resultin = []
    all_resultout = []
    for run_id in exp_ids:
        run_data = data[data[:, 0] == run_id]
        resultin = run_data[:, 1:1+num_qubits]
        resultout = run_data[:, 1+num_qubits:1+2*num_qubits]
        all_resultin.append(resultin)
        all_resultout.append(resultout)
    return all_resultin, all_resultout