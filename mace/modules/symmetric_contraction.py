###########################################################################################
# Implementation of the symmetric contraction algorithm presented in the MACE paper
# (Batatia et al, MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields , Eq.10 and 11)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, Optional, Union

import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode

from mace.tools.cg import U_matrix_real

import time

from torch.utils.dlpack import to_dlpack

import matplotlib.pyplot as plt
from scipy.sparse import random

BATCH_EXAMPLE = 5888
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


@compile_mode("script")
class SymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        del irreps_in, irreps_out

        if not isinstance(correlation, tuple):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleList()
        for irrep_out in self.irreps_out:
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outs = [contraction(x, y) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)


@compile_mode("script")
class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        dtype = torch.get_default_dtype()
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])

        for i in range(correlation, 0, -1):
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]

            print("cuEq Contraction irrep_out.lmax:%d, num_equivariance:%d " % (irrep_out.lmax,  num_equivariance))
            
            if i == correlation:
                parse_subscript_main = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum(
                        "".join(parse_subscript_main), x, y, w, z
                    )
                )

                print("========== einsum contract main %s ==========", "".join(parse_subscript_main))

                # Optimizing the contractions
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                

                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights_max = w
            else:
                # Generate optimized contractions equations
                parse_subscript_weighting = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    + ["k,ekc,be->bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    + ["i,bci->bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                )

                # Symbolic tracing of contractions
                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum(
                        "".join(parse_subscript_weighting), x, y, z
                    )
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )

                print("========== einsum contract weighting %s ==========", "".join(parse_subscript_weighting))
                print("========== einsum contract features %s ==========", "".join(parse_subscript_features))

                # Optimizing the contractions
                graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_weighting,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_features,
                    example_inputs=(
                        torch.randn(
                            [BATCH_EXAMPLE, self.num_features, num_equivariance]
                            + [num_ell] * i
                        ).squeeze(2),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                    ),
                )

                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights.append(w)
        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

    def stats_nonzero(self, tensor: torch.Tensor):
        nonzero_count = torch.count_nonzero(tensor)
        total_elements = tensor.numel()  # 或 tensor.size().numel()
        nonzero_ratio = nonzero_count / total_elements
        return nonzero_ratio

    def check_sparsity(self, sparse_tensor):
        nnz = torch.count_nonzero(sparse_tensor)
        total = sparse_tensor.numel()
        sparsity = 1 - (nnz / total)
        print(f"Sparsity: {sparsity:.2%} (nnz={nnz}, total={total})")
    
    '''
    def gen_sparse_matrix(self, dense_tensor):
        dense_cupy = cupy.fromDlpack(to_dlpack(dense_tensor))
        csr_matrix = cupyx.scipy.sparse.csr_matrix(dense_cupy)
        csr_matrix.sum_duplicates()
        csr_matrix.sort_indices()
        csr_matrix.has_canonical_format = True
        return csr_matrix
    '''

    def draw_tensor(self, sparse_matrix, name):
        plt.figure(figsize=(10, 10))
        plt.spy(sparse_matrix.cpu(), markersize=5)
        plt.title("Sparse Matrix Visualization")
        plt.savefig(name+'.png')  # 默认保存为PNG格式
        plt.close()  # 关闭图形，释放内存

    def draw_2d_tensor(self, tensor, name):
        merge = torch.zeros(16,16,device="cuda")
        for tid in range(tensor.shape[-1]):
            merge += tensor[:,:,tid]
        plt.figure(figsize=(10, 10))
        plt.spy(merge.cpu(), markersize=5)
        plt.title("Sparse Matrix Visualization")
        plt.savefig(name+'.png')  # 默认保存为PNG格式
        plt.close()  # 关闭图形，释放内存

    def use_spmm(self, x, y):
            # 1. 构建稀疏张量 [a, b, e, f, d]（COO 格式）
        a, b, e, f, d = y.shape
        a, b, d = x.shape
        sparse_5d = y.to_sparse_coo()
        indices = sparse_5d.indices()  # (5, nnz)
        values = sparse_5d.values()    # (nnz,)

        # 计算新的 2D 索引 [a*b*e*f, d]
        i, j, k, l, m = indices
        new_row = i * (b * e * f) + j * (e * f) + k * f + l
        new_col = m
        sparse_2d = torch.sparse_coo_tensor(
            torch.stack([new_row, new_col]),
            values,
            size=(a * b * e * f, d),
        ).coalesce().to_sparse_csr()

        # 2. 调整稠密张量 [a, b, d] -> [d, a*b]
        dense_3d = x
        dense_2d = dense_3d.reshape(a * b, d).T  # [a*b, d] -> [d, a*b]

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000
        # 3. 执行 SpMM: [a*b*e*f, d] @ [d, a*b] -> [a*b*e*f, a*b]
        result = torch.sparse.mm(sparse_2d, dense_2d)
        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"========= spmm only compute cost: {execution_time_ms:.3f} ms ========")
        # 4. 恢复形状（可选）
        final_result = result.reshape(a, b, e, f)
        print(final_result.shape)

    def diag_weight_einsum(self, A_tensor, B_tensor):
        W,X,K = A_tensor.shape
        B,K,C = B_tensor.shape
        A_diag = torch.stack([torch.diagonal(A_tensor[:, :, k]) for k in range(K)], dim=0)  # [K, W]

        torch.cuda.synchronize()
        C_diag = torch.einsum("kw,bkc->bcw", A_diag, B_tensor)  # [B, C, W]
        
        device = "cuda"
        # 构造全输出张量并写入对角
        C_opt = torch.zeros((B, C, W, X), device=device)
        w_idx = torch.arange(W, device=device)
        C_opt[:, :, w_idx, w_idx] = C_diag
        return C_opt

    def manual_weight_compute(self, A_tensor, B_tensor):
        device = "cuda"
        W,X,K = A_tensor.shape
        B,K,C = B_tensor.shape
        A_diag = torch.stack([torch.diagonal(A_tensor[:, :, k]) for k in range(K)], dim=0)  # [K, W]
        nonzero_map = {
            0: [0],
            1: [1, 2, 3],
            2: [4, 5, 6, 7, 8],
            3: [9, 10, 11, 12, 13, 14,15]
        }

        C_manual_full = torch.zeros(B, C, W, X, device=device)
        for k, ws in nonzero_map.items():
            for w in ws:
                #C_manual_full[:, :, w, w] += A_diag[k, w] * B_tensor[:, k, :]
                 C_manual_full[:, :, w, w] += A_tensor[w, w, k] * B_tensor[:, k, :]

        return C_manual_full

    
    def draw_tensor_3d(self, tensor: torch.Tensor, name: str):
        
        merge_tensor = torch.zeros(16,16,16, device="cuda")
        for tid in range(0, tensor.shape[-1]):
            merge_tensor += tensor[:,:,:,tid]
        
        nonzero = torch.nonzero(merge_tensor).cpu().numpy()
        
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))

        x_idx = 0
        yz_proj = torch.zeros(16, 16)
        for nonele in nonzero:
            if nonele[0] == x_idx:
                yz_proj[nonele[1], nonele[2]] = 1
            else:
                axes[x_idx/4, x_idx%4].imshow(yz_proj.numpy(), cmap='Greys')
                axes[x_idx/4, x_idx%4].set_title('YZ merge (non-zero)')
                axes[x_idx/4, x_idx%4].set_xlabel('Z')
                axes[x_idx/4, x_idx%4].set_ylabel('Y')
                yz_proj = torch.zeros(16, 16)
                x_idx = nonele[0]
                    
        plt.tight_layout()
        plt.savefig(str(name) + '.png', dpi=300)
        # draw 3d tensor
        '''
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制非零元素
        if len(nonzero_indices) > 0:
            x, y, z = nonzero_indices.T
            ax.scatter(x, y, z, c='r', marker='o', s=20, alpha=0.6, label='Non-zero elements')

        ax.set_xlabel('X (Dim 1)')
        ax.set_ylabel('Y (Dim 2)')
        ax.set_zlabel('Z (Dim 3)')
        ax.set_title('3D Tensor Non-Zero Elements Distribution')
        ax.legend()
        plt.savefig(str(name) + '.png', dpi=300)
        '''

    def print_nonzero_index(self, tensor: torch.Tensor):
        tensor_num = tensor.shape[-1]
        for i in range(0, tensor_num):
            tensor_i = tensor[:,:,:,i]
            print("<< tensor idx:%d >>" % i)
            nonzero_indices = torch.nonzero(tensor_i)
            for idx in nonzero_indices:
                print(idx.tolist(), tensor_i[idx[0], idx[1], idx[2]])

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000
        
        
        out = self.graph_opt_main(
            self.U_tensors(self.correlation),
            self.weights_max,
            x,
            y,
        )
        print("contract main out shape:", out.shape)
        
        #self.draw_tensor_3d(self.U_tensors(self.correlation), "cg3d")
        #self.print_nonzero_index(self.U_tensors(self.correlation))
        ''' 
        uw = torch.functional.tensordot(self.weights_max, self.U_tensors(self.correlation), dims = ((1,), (3,)), out = None) 
        y_idx = torch.argmax(y, dim=1)
        uw_selected = uw[y_idx]
        print("uw_selected shape:", uw_selected.shape)
        self.check_sparsity(uw_selected)
        print("x shape:", x.shape)
        #self.draw_tensor(uw_selected)
        out = torch.einsum('abd, abefd->abef', x, uw_selected)
        #out = self.use_spmm(x, uw_selected)
        '''
        y_idx = torch.argmax(y, dim=1)

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"========= total contract_main compute cost: {execution_time_ms:.3f} ms ========")        

        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            torch.cuda.synchronize()
            
            start_time = time.perf_counter() * 1000
            
            c_tensor = contract_weights(
                self.U_tensors(self.correlation - i - 1),
                weight,
                y,
            )


            if i == 0:
                torch.cuda.synchronize()
                start_time = time.perf_counter() * 1000
                w_selected = weight[y_idx]
                #c_tensor = self.diag_weight_einsum(self.U_tensors(self.correlation - i - 1), w_selected)
                c_tensor = self.manual_weight_compute(self.U_tensors(self.correlation - i - 1), w_selected)
                torch.cuda.synchronize()
                end_time = time.perf_counter() * 1000
                execution_time_ms = end_time - start_time
                print(f"========= my contract_weight cost: {execution_time_ms:.3f} ms ========")
                
                #c_tensor = c_tensor + out
                start_time = time.perf_counter() * 1000
                idx = torch.arange(c_tensor.shape[-1], device="cuda")
                c_diag = c_tensor[:, :, idx, idx]  # [N, C, W]
                out_diag = out[:, :, idx, idx]
                diag_sum = c_diag + out_diag
                out[:, :, idx, idx] = diag_sum
                torch.cuda.synchronize()
                end_time = time.perf_counter() * 1000
                execution_time_ms = end_time - start_time
                print(f"========= my tensor add cost: {execution_time_ms:.3f} ms ========")

                start_time = time.perf_counter() * 1000
                c_diag = c_tensor.diagonal(offset=0, dim1=2, dim2=3)  # [N, C, W]
                out = c_diag * x
                torch.cuda.synchronize()
                end_time = time.perf_counter() * 1000
                execution_time_ms = end_time - start_time
                print(f"========= my contract_features cost: {execution_time_ms:.3f} ms ========")

            else:
                start_time = time.perf_counter() * 1000
                c_tensor = contract_weights(
                    self.U_tensors(self.correlation - i - 1),
                    weight,
                    y,
                )
                
                torch.cuda.synchronize()
                end_time = time.perf_counter() * 1000
                execution_time_ms = end_time - start_time
                print(f"========= contract_weight cost: {execution_time_ms:.3f} ms ========")
                
                start_time = time.perf_counter() * 1000
                c_tensor = c_tensor + out
                print("contract weight out shape:", c_tensor.shape)
                torch.cuda.synchronize()
                end_time = time.perf_counter() * 1000
                execution_time_ms = end_time - start_time
                print(f"========= tensor add cost: {execution_time_ms:.3f} ms ========")

                start_time = time.perf_counter() * 1000
                out = contract_features(c_tensor, x)
                torch.cuda.synchronize()
                end_time = time.perf_counter() * 1000
                execution_time_ms = end_time - start_time
                print(f"========= contract_features cost: {execution_time_ms:.3f} ms ========")
                print("contract feature out shape:", out.shape)
        return out.view(out.shape[0], -1)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]
