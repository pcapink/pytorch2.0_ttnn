# High Level Operations Status
|    | Operations                     |   Input Variations |   Converted |   Removed |   Fallback | Completed   |   Score |
|---:|:-------------------------------|-------------------:|------------:|----------:|-----------:|:------------|--------:|
|  0 | aten._softmax.default          |                  4 |           4 |         0 |          0 | ✅          |    1    |
|  1 | aten._unsafe_view.default      |                 19 |          19 |         0 |          0 | ✅          |    1    |
|  2 | aten.add.Tensor                |                 11 |           8 |         0 |          0 | 🚧          |    0.73 |
|  3 | aten.addmm.default             |                 17 |          15 |         2 |          0 | ✅          |    1    |
|  4 | aten.as_strided.default        |                  1 |           1 |         0 |          0 | ✅          |    1    |
|  5 | aten.bmm.default               |                  8 |           8 |         0 |          0 | ✅          |    1    |
|  6 | aten.cat.default               |                  3 |           3 |         0 |          0 | ✅          |    1    |
|  7 | aten.clone.default             |                 39 |           0 |        39 |          0 | ✅          |    1    |
|  8 | aten.constant_pad_nd.default   |                  4 |           4 |         0 |          0 | ✅          |    1    |
|  9 | aten.convolution.default       |                  1 |           1 |         0 |          0 | ✅          |    1    |
| 10 | aten.eq.Scalar                 |                  3 |           3 |         0 |          0 | ✅          |    1    |
| 11 | aten.expand.default            |                 12 |           0 |        12 |          0 | ✅          |    1    |
| 12 | aten.fill.Tensor               |                 19 |           0 |         0 |          0 | ✘           |    0    |
| 13 | aten.gelu.default              |                  4 |           4 |         0 |          0 | ✅          |    1    |
| 14 | aten.index.Tensor              |                  4 |           0 |         4 |          0 | ✅          |    1    |
| 15 | aten.masked_fill.Scalar        |                  6 |           6 |         0 |          0 | ✅          |    1    |
| 16 | aten.mean.dim                  |                  1 |           1 |         0 |          0 | ✅          |    1    |
| 17 | aten.mm.default                |                  3 |           3 |         0 |          0 | ✅          |    1    |
| 18 | aten.mul.Tensor                |                  4 |           4 |         0 |          0 | ✅          |    1    |
| 19 | aten.native_layer_norm.default |                  7 |           7 |         0 |          0 | ✅          |    1    |
| 20 | aten.ne.Scalar                 |                  3 |           3 |         0 |          0 | ✅          |    1    |
| 21 | aten.new_zeros.default         |                  3 |           0 |         3 |          0 | ✅          |    1    |
| 22 | aten.permute.default           |                 21 |          17 |         4 |          0 | ✅          |    1    |
| 23 | aten.roll.default              |                  6 |           6 |         0 |          0 | ✅          |    1    |
| 24 | aten.select.int                |                 12 |          12 |         0 |          0 | ✅          |    1    |
| 25 | aten.slice.Tensor              |                 59 |          45 |        14 |          0 | ✅          |    1    |
| 26 | aten.slice_scatter.default     |                 36 |          36 |         0 |          0 | ✅          |    1    |
| 27 | aten.sub.Tensor                |                  3 |           3 |         0 |          0 | ✅          |    1    |
| 28 | aten.t.default                 |                 20 |           0 |        20 |          0 | ✅          |    1    |
| 29 | aten.transpose.int             |                  8 |           8 |         0 |          0 | ✅          |    1    |
| 30 | aten.unsqueeze.default         |                 16 |          12 |         4 |          0 | ✅          |    1    |
| 31 | aten.view.default              |                 73 |          69 |         4 |          0 | ✅          |    1    |
***
### aten._softmax.default
|    | ATen Input Variations                                                            | Status   | Isolated   |      PCC |   Host |
|---:|:---------------------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 24, 49, 49]> self = ?,<br>int dim = -1,<br>bool half_to_float = False | Done     | Done       | 0.999615 |      0 |
|  1 | Tensor<[16, 6, 49, 49]> self = ?,<br>int dim = -1,<br>bool half_to_float = False | Done     | Done       | 0.999609 |      0 |
|  2 | Tensor<[4, 12, 49, 49]> self = ?,<br>int dim = -1,<br>bool half_to_float = False | Done     | Done       | 0.999602 |      0 |
|  3 | Tensor<[64, 3, 49, 49]> self = ?,<br>int dim = -1,<br>bool half_to_float = False | Done     | Done       | 0.999604 |      0 |
### aten._unsafe_view.default
|    | ATen Input Variations                                                       | Status   | Isolated   |   PCC |   Host |
|---:|:----------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 2, 2, 7, 7, 384]> self = ?,<br>List[int] size = [4, 49, 384]     | Done     | Done       |     1 |      0 |
|  1 | Tensor<[1, 2, 7, 2, 7, 384]> self = ?,<br>List[int] size = [1, 14, 14, 384] | Done     | Done       |     1 |      0 |
|  2 | Tensor<[1, 4, 4, 7, 7, 192]> self = ?,<br>List[int] size = [16, 49, 192]    | Done     | Done       |     1 |      0 |
|  3 | Tensor<[1, 4, 7, 4, 7, 192]> self = ?,<br>List[int] size = [1, 28, 28, 192] | Done     | Done       |     1 |      0 |
|  4 | Tensor<[1, 49, 24, 32]> self = ?,<br>List[int] size = [1, 49, 768]          | Done     | Done       |     1 |      0 |
|  5 | Tensor<[1, 8, 7, 8, 7, 96]> self = ?,<br>List[int] size = [1, 56, 56, 96]   | Done     | Done       |     1 |      0 |
|  6 | Tensor<[1, 8, 8, 7, 7, 96]> self = ?,<br>List[int] size = [64, 49, 96]      | Done     | Done       |     1 |      0 |
|  7 | Tensor<[16, 49, 6, 32]> self = ?,<br>List[int] size = [16, 49, 192]         | Done     | Done       |     1 |      0 |
|  8 | Tensor<[16, 6, 32, 49]> self = ?,<br>List[int] size = [96, 32, 49]          | Done     | Done       |     1 |      0 |
|  9 | Tensor<[16, 6, 49, 32]> self = ?,<br>List[int] size = [96, 49, 32]          | Done     | Done       |     1 |      0 |
| 10 | Tensor<[2, 2, 7, 7]> self = ?,<br>List[int] size = [4, 49]                  | Done     | Done       |     1 |      0 |
| 11 | Tensor<[4, 12, 32, 49]> self = ?,<br>List[int] size = [48, 32, 49]          | Done     | Done       |     1 |      0 |
| 12 | Tensor<[4, 12, 49, 32]> self = ?,<br>List[int] size = [48, 49, 32]          | Done     | Done       |     1 |      0 |
| 13 | Tensor<[4, 4, 7, 7]> self = ?,<br>List[int] size = [16, 49]                 | Done     | Done       |     1 |      0 |
| 14 | Tensor<[4, 49, 12, 32]> self = ?,<br>List[int] size = [4, 49, 384]          | Done     | Done       |     1 |      0 |
| 15 | Tensor<[64, 3, 32, 49]> self = ?,<br>List[int] size = [192, 32, 49]         | Done     | Done       |     1 |      0 |
| 16 | Tensor<[64, 3, 49, 32]> self = ?,<br>List[int] size = [192, 49, 32]         | Done     | Done       |     1 |      0 |
| 17 | Tensor<[64, 49, 3, 32]> self = ?,<br>List[int] size = [64, 49, 96]          | Done     | Done       |     1 |      0 |
| 18 | Tensor<[8, 8, 7, 7]> self = ?,<br>List[int] size = [64, 49]                 | Done     | Done       |     1 |      0 |
### aten.add.Tensor
|    | ATen Input Variations                                                        | Status   | Isolated   |      PCC |   Host |
|---:|:-----------------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 14, 14, 384]> self = ?,<br>Tensor<[1, 14, 14, 384]> other = ?     | Done     | Done       | 0.999998 |      0 |
|  1 | Tensor<[1, 16, 6, 49, 49]> self = ?,<br>Tensor<[1, 16, 1, 49, 49]> other = ? | None     | Fallback   | 1        |     -1 |
|  2 | Tensor<[1, 24, 49, 49]> self = ?,<br>Tensor<[1, 24, 49, 49]> other = ?       | Done     | Done       | 0.999998 |      0 |
|  3 | Tensor<[1, 28, 28, 192]> self = ?,<br>Tensor<[1, 28, 28, 192]> other = ?     | Done     | Done       | 0.999998 |      0 |
|  4 | Tensor<[1, 4, 12, 49, 49]> self = ?,<br>Tensor<[1, 4, 1, 49, 49]> other = ?  | None     | Fallback   | 1        |     -1 |
|  5 | Tensor<[1, 56, 56, 96]> self = ?,<br>Tensor<[1, 56, 56, 96]> other = ?       | Done     | Done       | 0.999998 |      0 |
|  6 | Tensor<[1, 64, 3, 49, 49]> self = ?,<br>Tensor<[1, 64, 1, 49, 49]> other = ? | None     | Fallback   | 1        |     -1 |
|  7 | Tensor<[1, 7, 7, 768]> self = ?,<br>Tensor<[1, 7, 7, 768]> other = ?         | Done     | Done       | 0.999998 |      0 |
|  8 | Tensor<[16, 6, 49, 49]> self = ?,<br>Tensor<[1, 6, 49, 49]> other = ?        | Done     | Done       | 0.999998 |      0 |
|  9 | Tensor<[4, 12, 49, 49]> self = ?,<br>Tensor<[1, 12, 49, 49]> other = ?       | Done     | Done       | 0.999998 |      0 |
| 10 | Tensor<[64, 3, 49, 49]> self = ?,<br>Tensor<[1, 3, 49, 49]> other = ?        | Done     | Done       | 0.999998 |      0 |
### aten.addmm.default
|    | ATen Input Variations                                                                    | Status   | Isolated   |      PCC |   Host |
|---:|:-----------------------------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1000]> self = ?,<br>Tensor<[1, 768]> mat1 = ?,<br>Tensor<[768, 1000]> mat2 = ?   | Done     | Done       | 0.999967 |      0 |
|  1 | Tensor<[1152]> self = ?,<br>Tensor<[196, 384]> mat1 = ?,<br>Tensor<[384, 1152]> mat2 = ? | Done     | Done       | 0.999972 |      0 |
|  2 | Tensor<[1536]> self = ?,<br>Tensor<[196, 384]> mat1 = ?,<br>Tensor<[384, 1536]> mat2 = ? | Done     | Done       | 0.999971 |      0 |
|  3 | Tensor<[192]> self = ?,<br>Tensor<[784, 192]> mat1 = ?,<br>Tensor<[192, 192]> mat2 = ?   | Done     | Done       | 0.999976 |      0 |
|  4 | Tensor<[192]> self = ?,<br>Tensor<[784, 768]> mat1 = ?,<br>Tensor<[768, 192]> mat2 = ?   | Done     | Done       | 0.999967 |      0 |
|  5 | Tensor<[2304]> self = ?,<br>Tensor<[49, 768]> mat1 = ?,<br>Tensor<[768, 2304]> mat2 = ?  | Removed  | Done       | 0.999967 |      0 |
|  6 | Tensor<[288]> self = ?,<br>Tensor<[3136, 96]> mat1 = ?,<br>Tensor<[96, 288]> mat2 = ?    | Done     | Done       | 0.999983 |      0 |
|  7 | Tensor<[3072]> self = ?,<br>Tensor<[49, 768]> mat1 = ?,<br>Tensor<[768, 3072]> mat2 = ?  | Done     | Done       | 0.999967 |      0 |
|  8 | Tensor<[384]> self = ?,<br>Tensor<[196, 1536]> mat1 = ?,<br>Tensor<[1536, 384]> mat2 = ? | Done     | Done       | 0.999937 |      0 |
|  9 | Tensor<[384]> self = ?,<br>Tensor<[196, 384]> mat1 = ?,<br>Tensor<[384, 384]> mat2 = ?   | Done     | Done       | 0.999975 |      0 |
| 10 | Tensor<[384]> self = ?,<br>Tensor<[3136, 96]> mat1 = ?,<br>Tensor<[96, 384]> mat2 = ?    | Done     | Done       | 0.999983 |      0 |
| 11 | Tensor<[576]> self = ?,<br>Tensor<[784, 192]> mat1 = ?,<br>Tensor<[192, 576]> mat2 = ?   | Done     | Done       | 0.99998  |      0 |
| 12 | Tensor<[768]> self = ?,<br>Tensor<[49, 3072]> mat1 = ?,<br>Tensor<[3072, 768]> mat2 = ?  | Done     | Done       | 0.999944 |      0 |
| 13 | Tensor<[768]> self = ?,<br>Tensor<[49, 768]> mat1 = ?,<br>Tensor<[768, 768]> mat2 = ?    | Removed  | Done       | 0.999967 |      0 |
| 14 | Tensor<[768]> self = ?,<br>Tensor<[784, 192]> mat1 = ?,<br>Tensor<[192, 768]> mat2 = ?   | Done     | Done       | 0.99998  |      0 |
| 15 | Tensor<[96]> self = ?,<br>Tensor<[3136, 384]> mat1 = ?,<br>Tensor<[384, 96]> mat2 = ?    | Done     | Done       | 0.999972 |      0 |
| 16 | Tensor<[96]> self = ?,<br>Tensor<[3136, 96]> mat1 = ?,<br>Tensor<[96, 96]> mat2 = ?      | Done     | Done       | 0.999983 |      0 |
### aten.as_strided.default
|    | ATen Input Variations                                                                                         | Status   | Isolated   |   PCC |   Host |
|---:|:--------------------------------------------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 768, 1, 1]> self = ?,<br>List[int] size = [1, 768, 1, 1],<br>List[int] stride = [768, 1, 768, 768] | Done     | Done       |     1 |      0 |
### aten.bmm.default
|    | ATen Input Variations                                             | Status   | Isolated   |      PCC |   Host |
|---:|:------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[192, 49, 32]> self = ?,<br>Tensor<[192, 32, 49]> mat2 = ? | Done     | Done       | 0.999989 |      0 |
|  1 | Tensor<[192, 49, 49]> self = ?,<br>Tensor<[192, 49, 32]> mat2 = ? | Done     | Done       | 0.999986 |      0 |
|  2 | Tensor<[24, 49, 32]> self = ?,<br>Tensor<[24, 32, 49]> mat2 = ?   | Done     | Done       | 0.99999  |      0 |
|  3 | Tensor<[24, 49, 49]> self = ?,<br>Tensor<[24, 49, 32]> mat2 = ?   | Done     | Done       | 0.999986 |      0 |
|  4 | Tensor<[48, 49, 32]> self = ?,<br>Tensor<[48, 32, 49]> mat2 = ?   | Done     | Done       | 0.99999  |      0 |
|  5 | Tensor<[48, 49, 49]> self = ?,<br>Tensor<[48, 49, 32]> mat2 = ?   | Done     | Done       | 0.999987 |      0 |
|  6 | Tensor<[96, 49, 32]> self = ?,<br>Tensor<[96, 32, 49]> mat2 = ?   | Done     | Done       | 0.999989 |      0 |
|  7 | Tensor<[96, 49, 49]> self = ?,<br>Tensor<[96, 49, 32]> mat2 = ?   | Done     | Done       | 0.999986 |      0 |
### aten.cat.default
|    | ATen Input Variations                                                                                                    | Status   | Isolated   |   PCC |   Host |
|---:|:-------------------------------------------------------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | List[Tensor] tensors = [<[1, 14, 14, 192]>, <[1, 14, 14, 192]>, <[1, 14, 14, 192]>, <[1, 14, 14, 192]>],<br>int dim = -1 | Done     | Done       |     1 |      0 |
|  1 | List[Tensor] tensors = [<[1, 28, 28, 96]>, <[1, 28, 28, 96]>, <[1, 28, 28, 96]>, <[1, 28, 28, 96]>],<br>int dim = -1     | Done     | Done       |     1 |      0 |
|  2 | List[Tensor] tensors = [<[1, 7, 7, 384]>, <[1, 7, 7, 384]>, <[1, 7, 7, 384]>, <[1, 7, 7, 384]>],<br>int dim = -1         | Done     | Done       |     1 |      0 |
### aten.clone.default
|    | ATen Input Variations                                                                           | Status   | Isolated   | PCC   | Host   |
|---:|:------------------------------------------------------------------------------------------------|:---------|:-----------|:------|:-------|
|  0 | Tensor<[1, 14, 14, 1536]> self = ?                                                              | Removed  | Done       | 1.0   | 0      |
|  1 | Tensor<[1, 14, 14, 384]> self = ?                                                               | Removed  | Done       | 1.0   | 0      |
|  2 | Tensor<[1, 2, 2, 7, 7, 384]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format | Removed  | Done       | 1.0   | 0      |
|  3 | Tensor<[1, 2, 7, 2, 7, 384]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format | Removed  | Done       | 1.0   | 0      |
|  4 | Tensor<[1, 24, 49, 49]> self = ?                                                                | Removed  | Done       | 1.0   | 0      |
|  5 | Tensor<[1, 28, 28, 192]> self = ?                                                               | Removed  | Done       | 1.0   | 0      |
|  6 | Tensor<[1, 28, 28, 768]> self = ?                                                               | Removed  | Done       | 1.0   | 0      |
|  7 | Tensor<[1, 4, 4, 7, 7, 192]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format | Removed  | Done       | 1.0   | 0      |
|  8 | Tensor<[1, 4, 7, 4, 7, 192]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format | Removed  | Done       | 1.0   | 0      |
|  9 | Tensor<[1, 49, 24, 32]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format      | Removed  | Done       | 1.0   | 0      |
| 10 | Tensor<[1, 49, 768]> self = ?                                                                   | Removed  | Done       | 1.0   | 0      |
| 11 | Tensor<[1, 56, 56, 384]> self = ?                                                               | Removed  | Done       | 1.0   | 0      |
| 12 | Tensor<[1, 56, 56, 96]> self = ?                                                                | Removed  | Done       | 1.0   | 0      |
| 13 | Tensor<[1, 7, 7, 3072]> self = ?                                                                | Removed  | Done       | 1.0   | 0      |
| 14 | Tensor<[1, 7, 7, 768]> self = ?                                                                 | Removed  | Done       | 1.0   | 0      |
| 15 | Tensor<[1, 8, 7, 8, 7, 96]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format  | Removed  | Done       | 1.0   | 0      |
| 16 | Tensor<[1, 8, 8, 7, 7, 96]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format  | Removed  | Unknown    | N/A   | N/A    |
| 17 | Tensor<[12, 49, 49]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format         | Removed  | Done       | 1.0   | 0      |
| 18 | Tensor<[16, 49, 192]> self = ?                                                                  | Removed  | Done       | 1.0   | 0      |
| 19 | Tensor<[16, 49, 6, 32]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format      | Removed  | Done       | 1.0   | 0      |
| 20 | Tensor<[16, 6, 32, 49]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format      | Removed  | Done       | 1.0   | 0      |
| 21 | Tensor<[16, 6, 49, 32]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format      | Removed  | Done       | 1.0   | 0      |
| 22 | Tensor<[16, 6, 49, 49]> self = ?                                                                | Removed  | Done       | 1.0   | 0      |
| 23 | Tensor<[2, 2, 7, 7]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format         | Removed  | Done       | 1.0   | 0      |
| 24 | Tensor<[24, 49, 49]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format         | Removed  | Done       | 1.0   | 0      |
| 25 | Tensor<[3, 49, 49]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format          | Removed  | Unknown    | N/A   | N/A    |
| 26 | Tensor<[4, 12, 32, 49]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format      | Removed  | Done       | 1.0   | 0      |
| 27 | Tensor<[4, 12, 49, 32]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format      | Removed  | Done       | 1.0   | 0      |
| 28 | Tensor<[4, 12, 49, 49]> self = ?                                                                | Removed  | Done       | 1.0   | 0      |
| 29 | Tensor<[4, 4, 7, 7]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format         | Removed  | Done       | 1.0   | 0      |
| 30 | Tensor<[4, 49, 12, 32]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format      | Removed  | Done       | 1.0   | 0      |
| 31 | Tensor<[4, 49, 384]> self = ?                                                                   | Removed  | Done       | 1.0   | 0      |
| 32 | Tensor<[6, 49, 49]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format          | Removed  | Done       | 1.0   | 0      |
| 33 | Tensor<[64, 3, 32, 49]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format      | Removed  | Unknown    | N/A   | N/A    |
| 34 | Tensor<[64, 3, 49, 32]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format      | Removed  | Unknown    | N/A   | N/A    |
| 35 | Tensor<[64, 3, 49, 49]> self = ?                                                                | Removed  | Unknown    | N/A   | N/A    |
| 36 | Tensor<[64, 49, 3, 32]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format      | Removed  | Done       | 1.0   | 0      |
| 37 | Tensor<[64, 49, 96]> self = ?                                                                   | Removed  | Done       | 1.0   | 0      |
| 38 | Tensor<[8, 8, 7, 7]> self = ?,<br>Optional[int] memory_format = torch.contiguous_format         | Removed  | Done       | 1.0   | 0      |
### aten.constant_pad_nd.default
|    | ATen Input Variations                                                                           | Status   | Isolated   |   PCC |   Host |
|---:|:------------------------------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 14, 14, 384]> self = ?,<br>List[int] pad = [0, 0, 0, 0, 0, 0],<br>number value = 0.0 | Done     | Done       |     1 |      0 |
|  1 | Tensor<[1, 28, 28, 192]> self = ?,<br>List[int] pad = [0, 0, 0, 0, 0, 0],<br>number value = 0.0 | Done     | Done       |     1 |      0 |
|  2 | Tensor<[1, 56, 56, 96]> self = ?,<br>List[int] pad = [0, 0, 0, 0, 0, 0],<br>number value = 0.0  | Done     | Done       |     1 |      0 |
|  3 | Tensor<[1, 7, 7, 768]> self = ?,<br>List[int] pad = [0, 0, 0, 0, 0, 0],<br>number value = 0.0   | Done     | Done       |     1 |      0 |
### aten.convolution.default
|    | ATen Input Variations                                                                                                                                                                                                                                                                         | Status   | Isolated   |      PCC |   Host |
|---:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 3, 224, 224]> input = ?,<br>Tensor<[96, 3, 4, 4]> weight = ?,<br>Optional[Tensor]<[96]> bias = ?,<br>List[int] stride = [4, 4],<br>List[int] padding = [0, 0],<br>List[int] dilation = [1, 1],<br>bool transposed = False,<br>List[int] output_padding = [0, 0],<br>int groups = 1 | Done     | Done       | 0.999968 |      1 |
### aten.eq.Scalar
|    | ATen Input Variations                              | Status   | Isolated   |   PCC |   Host |
|---:|:---------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[16, 49, 49]> self = ?,<br>number other = 0 | Done     | Done       |     1 |      0 |
|  1 | Tensor<[4, 49, 49]> self = ?,<br>number other = 0  | Done     | Done       |     1 |      0 |
|  2 | Tensor<[64, 49, 49]> self = ?,<br>number other = 0 | Done     | Done       |     1 |      0 |
### aten.expand.default
|    | ATen Input Variations                                                 | Status   | Isolated   |   PCC |   Host |
|---:|:----------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 24, 32, 49]> self = ?,<br>List[int] size = [1, 24, 32, 49] | Removed  | Done       |     1 |     -1 |
|  1 | Tensor<[1, 24, 49, 32]> self = ?,<br>List[int] size = [1, 24, 49, 32] | Removed  | Done       |     1 |     -1 |
|  2 | Tensor<[1, 24, 49, 49]> self = ?,<br>List[int] size = [1, 24, 49, 49] | Removed  | Done       |     1 |     -1 |
|  3 | Tensor<[16, 6, 32, 49]> self = ?,<br>List[int] size = [16, 6, 32, 49] | Removed  | Done       |     1 |     -1 |
|  4 | Tensor<[16, 6, 49, 32]> self = ?,<br>List[int] size = [16, 6, 49, 32] | Removed  | Done       |     1 |     -1 |
|  5 | Tensor<[16, 6, 49, 49]> self = ?,<br>List[int] size = [16, 6, 49, 49] | Removed  | Done       |     1 |     -1 |
|  6 | Tensor<[4, 12, 32, 49]> self = ?,<br>List[int] size = [4, 12, 32, 49] | Removed  | Done       |     1 |     -1 |
|  7 | Tensor<[4, 12, 49, 32]> self = ?,<br>List[int] size = [4, 12, 49, 32] | Removed  | Done       |     1 |     -1 |
|  8 | Tensor<[4, 12, 49, 49]> self = ?,<br>List[int] size = [4, 12, 49, 49] | Removed  | Done       |     1 |     -1 |
|  9 | Tensor<[64, 3, 32, 49]> self = ?,<br>List[int] size = [64, 3, 32, 49] | Removed  | Done       |     1 |     -1 |
| 10 | Tensor<[64, 3, 49, 32]> self = ?,<br>List[int] size = [64, 3, 49, 32] | Removed  | Done       |     1 |     -1 |
| 11 | Tensor<[64, 3, 49, 49]> self = ?,<br>List[int] size = [64, 3, 49, 49] | Removed  | Done       |     1 |     -1 |
### aten.fill.Tensor
|    | ATen Input Variations                          | Status   | Isolated   | PCC   | Host   |
|---:|:-----------------------------------------------|:---------|:-----------|:------|:-------|
|  0 | Tensor<[21, 21]> self = ?,<br>Tensor value = ? | None     | Unknown    | N/A   | N/A    |
|  1 | Tensor<[21, 3]> self = ?,<br>Tensor value = ?  | None     | Unknown    | N/A   | N/A    |
|  2 | Tensor<[21, 4]> self = ?,<br>Tensor value = ?  | None     | Unknown    | N/A   | N/A    |
|  3 | Tensor<[3, 21]> self = ?,<br>Tensor value = ?  | None     | Unknown    | N/A   | N/A    |
|  4 | Tensor<[3, 3]> self = ?,<br>Tensor value = ?   | None     | Unknown    | N/A   | N/A    |
|  5 | Tensor<[3, 49]> self = ?,<br>Tensor value = ?  | None     | Unknown    | N/A   | N/A    |
|  6 | Tensor<[3, 4]> self = ?,<br>Tensor value = ?   | None     | Unknown    | N/A   | N/A    |
|  7 | Tensor<[3, 7]> self = ?,<br>Tensor value = ?   | None     | Unknown    | N/A   | N/A    |
|  8 | Tensor<[4, 21]> self = ?,<br>Tensor value = ?  | None     | Unknown    | N/A   | N/A    |
|  9 | Tensor<[4, 3]> self = ?,<br>Tensor value = ?   | None     | Unknown    | N/A   | N/A    |
| 10 | Tensor<[4, 49]> self = ?,<br>Tensor value = ?  | None     | Unknown    | N/A   | N/A    |
| 11 | Tensor<[4, 4]> self = ?,<br>Tensor value = ?   | None     | Unknown    | N/A   | N/A    |
| 12 | Tensor<[4, 7]> self = ?,<br>Tensor value = ?   | None     | Unknown    | N/A   | N/A    |
| 13 | Tensor<[49, 3]> self = ?,<br>Tensor value = ?  | None     | Unknown    | N/A   | N/A    |
| 14 | Tensor<[49, 49]> self = ?,<br>Tensor value = ? | None     | Unknown    | N/A   | N/A    |
| 15 | Tensor<[49, 4]> self = ?,<br>Tensor value = ?  | None     | Unknown    | N/A   | N/A    |
| 16 | Tensor<[7, 3]> self = ?,<br>Tensor value = ?   | None     | Unknown    | N/A   | N/A    |
| 17 | Tensor<[7, 4]> self = ?,<br>Tensor value = ?   | None     | Unknown    | N/A   | N/A    |
| 18 | Tensor<[7, 7]> self = ?,<br>Tensor value = ?   | None     | Unknown    | N/A   | N/A    |
### aten.gelu.default
|    | ATen Input Variations              | Status   | Isolated   |      PCC |   Host |
|---:|:-----------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 14, 14, 1536]> self = ? | Done     | Done       | 0.999991 |      0 |
|  1 | Tensor<[1, 28, 28, 768]> self = ?  | Done     | Done       | 0.999991 |      0 |
|  2 | Tensor<[1, 56, 56, 384]> self = ?  | Done     | Done       | 0.999991 |      0 |
|  3 | Tensor<[1, 7, 7, 3072]> self = ?   | Done     | Done       | 0.999991 |      0 |
### aten.index.Tensor
|    | ATen Input Variations                                                      | Status   | Isolated   |   PCC |   Host |
|---:|:---------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[169, 12]> self = ?,<br>List[Optional[Tensor]] indices = [<[2401]>] | Removed  | Done       |     1 |      0 |
|  1 | Tensor<[169, 24]> self = ?,<br>List[Optional[Tensor]] indices = [<[2401]>] | Removed  | Done       |     1 |      0 |
|  2 | Tensor<[169, 3]> self = ?,<br>List[Optional[Tensor]] indices = [<[2401]>]  | Removed  | Done       |     1 |      0 |
|  3 | Tensor<[169, 6]> self = ?,<br>List[Optional[Tensor]] indices = [<[2401]>]  | Removed  | Done       |     1 |      0 |
### aten.masked_fill.Scalar
|    | ATen Input Variations                                                                     | Status   | Isolated   |   PCC |   Host |
|---:|:------------------------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[16, 49, 49]> self = ?,<br>Tensor<[16, 49, 49]> mask = ?,<br>number value = -100.0 | Done     | Done       |     1 |      0 |
|  1 | Tensor<[16, 49, 49]> self = ?,<br>Tensor<[16, 49, 49]> mask = ?,<br>number value = 0.0    | Done     | Done       |     1 |      0 |
|  2 | Tensor<[4, 49, 49]> self = ?,<br>Tensor<[4, 49, 49]> mask = ?,<br>number value = -100.0   | Done     | Done       |     1 |      0 |
|  3 | Tensor<[4, 49, 49]> self = ?,<br>Tensor<[4, 49, 49]> mask = ?,<br>number value = 0.0      | Done     | Done       |     1 |      0 |
|  4 | Tensor<[64, 49, 49]> self = ?,<br>Tensor<[64, 49, 49]> mask = ?,<br>number value = -100.0 | Done     | Done       |     1 |      0 |
|  5 | Tensor<[64, 49, 49]> self = ?,<br>Tensor<[64, 49, 49]> mask = ?,<br>number value = 0.0    | Done     | Done       |     1 |      0 |
### aten.mean.dim
|    | ATen Input Variations                                                                          | Status   | Isolated   |      PCC |   Host |
|---:|:-----------------------------------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 768, 7, 7]> self = ?,<br>Optional[List[int]] dim = [-1, -2],<br>bool keepdim = True | Done     | Done       | 0.999997 |      0 |
### aten.mm.default
|    | ATen Input Variations                                        | Status   | Isolated   |      PCC |   Host |
|---:|:-------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[196, 768]> self = ?,<br>Tensor<[768, 384]> mat2 = ?  | Done     | Done       | 0.999963 |      0 |
|  1 | Tensor<[49, 1536]> self = ?,<br>Tensor<[1536, 768]> mat2 = ? | Done     | Done       | 0.999961 |      0 |
|  2 | Tensor<[784, 384]> self = ?,<br>Tensor<[384, 192]> mat2 = ?  | Done     | Done       | 0.999973 |      0 |
### aten.mul.Tensor
|    | ATen Input Variations                                                  | Status   | Isolated   |      PCC |   Host |
|---:|:-----------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 24, 49, 32]> self = ?,<br>Tensor other = 0.1767766952966369 | Done     | Done       | 0.999996 |      0 |
|  1 | Tensor<[16, 6, 49, 32]> self = ?,<br>Tensor other = 0.1767766952966369 | Done     | Done       | 0.999996 |      0 |
|  2 | Tensor<[4, 12, 49, 32]> self = ?,<br>Tensor other = 0.1767766952966369 | Done     | Done       | 0.999996 |      0 |
|  3 | Tensor<[64, 3, 49, 32]> self = ?,<br>Tensor other = 0.1767766952966369 | Done     | Done       | 0.999996 |      0 |
### aten.native_layer_norm.default
|    | ATen Input Variations                                                                                                                                                         | Status   | Isolated   |      PCC |   Host |
|---:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[1, 14, 14, 384]> input = ?,<br>List[int] normalized_shape = [384],<br>Optional[Tensor]<[384]> weight = ?,<br>Optional[Tensor]<[384]> bias = ?,<br>float eps = 1e-05   | Done     | Done       | 0.998762 |      3 |
|  1 | Tensor<[1, 14, 14, 768]> input = ?,<br>List[int] normalized_shape = [768],<br>Optional[Tensor]<[768]> weight = ?,<br>Optional[Tensor]<[768]> bias = ?,<br>float eps = 1e-05   | Done     | Done       | 0.997342 |      3 |
|  2 | Tensor<[1, 28, 28, 192]> input = ?,<br>List[int] normalized_shape = [192],<br>Optional[Tensor]<[192]> weight = ?,<br>Optional[Tensor]<[192]> bias = ?,<br>float eps = 1e-05   | Done     | Done       | 0.99933  |      3 |
|  3 | Tensor<[1, 28, 28, 384]> input = ?,<br>List[int] normalized_shape = [384],<br>Optional[Tensor]<[384]> weight = ?,<br>Optional[Tensor]<[384]> bias = ?,<br>float eps = 1e-05   | Done     | Done       | 0.998541 |      3 |
|  4 | Tensor<[1, 56, 56, 96]> input = ?,<br>List[int] normalized_shape = [96],<br>Optional[Tensor]<[96]> weight = ?,<br>Optional[Tensor]<[96]> bias = ?,<br>float eps = 1e-05       | Done     | Done       | 0.999666 |      3 |
|  5 | Tensor<[1, 7, 7, 1536]> input = ?,<br>List[int] normalized_shape = [1536],<br>Optional[Tensor]<[1536]> weight = ?,<br>Optional[Tensor]<[1536]> bias = ?,<br>float eps = 1e-05 | Done     | Done       | 0.991475 |      3 |
|  6 | Tensor<[1, 7, 7, 768]> input = ?,<br>List[int] normalized_shape = [768],<br>Optional[Tensor]<[768]> weight = ?,<br>Optional[Tensor]<[768]> bias = ?,<br>float eps = 1e-05     | Done     | Done       | 0.99737  |      3 |
### aten.ne.Scalar
|    | ATen Input Variations                              | Status   | Isolated   |   PCC |   Host |
|---:|:---------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[16, 49, 49]> self = ?,<br>number other = 0 | Done     | Done       |     1 |      0 |
|  1 | Tensor<[4, 49, 49]> self = ?,<br>number other = 0  | Done     | Done       |     1 |      0 |
|  2 | Tensor<[64, 49, 49]> self = ?,<br>number other = 0 | Done     | Done       |     1 |      0 |
### aten.new_zeros.default
|    | ATen Input Variations                                                                              | Status   | Isolated   |   PCC |   Host |
|---:|:---------------------------------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[16, 49, 192]> self = ?,<br>List[int] size = [28, 28],<br>Optional[bool] pin_memory = False | Removed  | Done       |     1 |      0 |
|  1 | Tensor<[4, 49, 384]> self = ?,<br>List[int] size = [14, 14],<br>Optional[bool] pin_memory = False  | Removed  | Done       |     1 |      0 |
|  2 | Tensor<[64, 49, 96]> self = ?,<br>List[int] size = [56, 56],<br>Optional[bool] pin_memory = False  | Removed  | Done       |     1 |      0 |
### aten.permute.default
|    | ATen Input Variations                                                         | Status   | Isolated   |   PCC |   Host |
|---:|:------------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 1, 1, 7, 7, 768]> self = ?,<br>List[int] dims = [0, 1, 3, 2, 4, 5] | Done     | Done       |     1 |      0 |
|  1 | Tensor<[1, 1, 7, 1, 7, 768]> self = ?,<br>List[int] dims = [0, 1, 3, 2, 4, 5] | Done     | Done       |     1 |      0 |
|  2 | Tensor<[1, 2, 2, 7, 7, 384]> self = ?,<br>List[int] dims = [0, 1, 3, 2, 4, 5] | Done     | Done       |     1 |      0 |
|  3 | Tensor<[1, 2, 7, 2, 7, 384]> self = ?,<br>List[int] dims = [0, 1, 3, 2, 4, 5] | Done     | Done       |     1 |      0 |
|  4 | Tensor<[1, 4, 4, 7, 7, 192]> self = ?,<br>List[int] dims = [0, 1, 3, 2, 4, 5] | Done     | Done       |     1 |      0 |
|  5 | Tensor<[1, 4, 7, 4, 7, 192]> self = ?,<br>List[int] dims = [0, 1, 3, 2, 4, 5] | Done     | Done       |     1 |      0 |
|  6 | Tensor<[1, 49, 3, 24, 32]> self = ?,<br>List[int] dims = [2, 0, 3, 1, 4]      | Done     | Done       |     1 |      0 |
|  7 | Tensor<[1, 7, 7, 768]> self = ?,<br>List[int] dims = [0, 3, 1, 2]             | Done     | Done       |     1 |      0 |
|  8 | Tensor<[1, 8, 7, 8, 7, 96]> self = ?,<br>List[int] dims = [0, 1, 3, 2, 4, 5]  | Done     | Done       |     1 |      0 |
|  9 | Tensor<[1, 8, 8, 7, 7, 96]> self = ?,<br>List[int] dims = [0, 1, 3, 2, 4, 5]  | Done     | Done       |     1 |      0 |
| 10 | Tensor<[1, 96, 56, 56]> self = ?,<br>List[int] dims = [0, 2, 3, 1]            | Done     | Done       |     1 |      0 |
| 11 | Tensor<[16, 49, 3, 6, 32]> self = ?,<br>List[int] dims = [2, 0, 3, 1, 4]      | Done     | Done       |     1 |      0 |
| 12 | Tensor<[2, 7, 2, 7]> self = ?,<br>List[int] dims = [0, 2, 1, 3]               | Done     | Done       |     1 |      0 |
| 13 | Tensor<[4, 49, 3, 12, 32]> self = ?,<br>List[int] dims = [2, 0, 3, 1, 4]      | Done     | Done       |     1 |      0 |
| 14 | Tensor<[4, 7, 4, 7]> self = ?,<br>List[int] dims = [0, 2, 1, 3]               | Done     | Done       |     1 |      0 |
| 15 | Tensor<[49, 49, 12]> self = ?,<br>List[int] dims = [2, 0, 1]                  | Removed  | Done       |     1 |      0 |
| 16 | Tensor<[49, 49, 24]> self = ?,<br>List[int] dims = [2, 0, 1]                  | Removed  | Done       |     1 |      0 |
| 17 | Tensor<[49, 49, 3]> self = ?,<br>List[int] dims = [2, 0, 1]                   | Removed  | Done       |     1 |      0 |
| 18 | Tensor<[49, 49, 6]> self = ?,<br>List[int] dims = [2, 0, 1]                   | Removed  | Done       |     1 |      0 |
| 19 | Tensor<[64, 49, 3, 3, 32]> self = ?,<br>List[int] dims = [2, 0, 3, 1, 4]      | Done     | Done       |     1 |      0 |
| 20 | Tensor<[8, 7, 8, 7]> self = ?,<br>List[int] dims = [0, 2, 1, 3]               | Done     | Done       |     1 |      0 |
### aten.roll.default
|    | ATen Input Variations                                                                         | Status   | Isolated   |   PCC |   Host |
|---:|:----------------------------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 14, 14, 384]> self = ?,<br>List[int] shifts = [-3, -3],<br>List[int] dims = [1, 2] | Done     | Done       |     1 |      0 |
|  1 | Tensor<[1, 14, 14, 384]> self = ?,<br>List[int] shifts = [3, 3],<br>List[int] dims = [1, 2]   | Done     | Done       |     1 |      0 |
|  2 | Tensor<[1, 28, 28, 192]> self = ?,<br>List[int] shifts = [-3, -3],<br>List[int] dims = [1, 2] | Done     | Done       |     1 |      0 |
|  3 | Tensor<[1, 28, 28, 192]> self = ?,<br>List[int] shifts = [3, 3],<br>List[int] dims = [1, 2]   | Done     | Done       |     1 |      0 |
|  4 | Tensor<[1, 56, 56, 96]> self = ?,<br>List[int] shifts = [-3, -3],<br>List[int] dims = [1, 2]  | Done     | Done       |     1 |      0 |
|  5 | Tensor<[1, 56, 56, 96]> self = ?,<br>List[int] shifts = [3, 3],<br>List[int] dims = [1, 2]    | Done     | Done       |     1 |      0 |
### aten.select.int
|    | ATen Input Variations                                                 | Status   | Isolated   |   PCC |   Host |
|---:|:----------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[3, 1, 24, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 0 | Done     | Done       |     1 |      0 |
|  1 | Tensor<[3, 1, 24, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 1 | Done     | Done       |     1 |      0 |
|  2 | Tensor<[3, 1, 24, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 2 | Done     | Done       |     1 |      0 |
|  3 | Tensor<[3, 16, 6, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 0 | Done     | Done       |     1 |      0 |
|  4 | Tensor<[3, 16, 6, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 1 | Done     | Done       |     1 |      0 |
|  5 | Tensor<[3, 16, 6, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 2 | Done     | Done       |     1 |      0 |
|  6 | Tensor<[3, 4, 12, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 0 | Done     | Done       |     1 |      0 |
|  7 | Tensor<[3, 4, 12, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 1 | Done     | Done       |     1 |      0 |
|  8 | Tensor<[3, 4, 12, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 2 | Done     | Done       |     1 |      0 |
|  9 | Tensor<[3, 64, 3, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 0 | Done     | Done       |     1 |      0 |
| 10 | Tensor<[3, 64, 3, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 1 | Done     | Done       |     1 |      0 |
| 11 | Tensor<[3, 64, 3, 49, 32]> self = ?,<br>int dim = 0,<br>int index = 2 | Done     | Done       |     1 |      0 |
### aten.slice.Tensor
|    | ATen Input Variations                                                                                                                      | Status   | Isolated   | PCC   | Host   |
|---:|:-------------------------------------------------------------------------------------------------------------------------------------------|:---------|:-----------|:------|:-------|
|  0 | Tensor<[1, 14, 14, 192]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
|  1 | Tensor<[1, 14, 14, 384]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
|  2 | Tensor<[1, 14, 14, 384]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
|  3 | Tensor<[1, 14, 14, 384]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
|  4 | Tensor<[1, 14, 14, 384]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
|  5 | Tensor<[1, 14, 28, 192]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
|  6 | Tensor<[1, 14, 28, 192]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
|  7 | Tensor<[1, 28, 28, 192]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
|  8 | Tensor<[1, 28, 28, 192]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
|  9 | Tensor<[1, 28, 28, 192]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2 | Done     | Done       | 1.0   | 0      |
| 10 | Tensor<[1, 28, 28, 192]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                  | Removed  | Done       | 1.0   | -1     |
| 11 | Tensor<[1, 28, 28, 96]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 12 | Tensor<[1, 28, 56, 96]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 13 | Tensor<[1, 28, 56, 96]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 14 | Tensor<[1, 56, 56, 96]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 15 | Tensor<[1, 56, 56, 96]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 16 | Tensor<[1, 56, 56, 96]> self = ?,<br>int dim = 1,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 17 | Tensor<[1, 56, 56, 96]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                   | Removed  | Done       | 1.0   | -1     |
| 18 | Tensor<[1, 7, 14, 384]> self = ?,<br>int dim = 2,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 19 | Tensor<[1, 7, 14, 384]> self = ?,<br>int dim = 2,<br>Optional[int] start = 1,<br>Optional[int] end = 9223372036854775807,<br>int step = 2  | Done     | Done       | 1.0   | 0      |
| 20 | Tensor<[1, 7, 7, 384]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 21 | Tensor<[1, 7, 7, 768]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 22 | Tensor<[1, 7, 7, 768]> self = ?,<br>int dim = 3,<br>Optional[int] start = 0,<br>Optional[int] end = 9223372036854775807                    | Removed  | Done       | 1.0   | -1     |
| 23 | Tensor<[14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 24 | Tensor<[14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                          | Done     | Unknown    | N/A   | N/A    |
| 25 | Tensor<[14, 14]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                           | Done     | Unknown    | N/A   | N/A    |
| 26 | Tensor<[21, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 27 | Tensor<[21, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                          | Done     | Unknown    | N/A   | N/A    |
| 28 | Tensor<[21, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                           | Removed  | Unknown    | N/A   | N/A    |
| 29 | Tensor<[28, 28]> self = ?,<br>int dim = 0,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 30 | Tensor<[28, 28]> self = ?,<br>int dim = 0,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                          | Done     | Unknown    | N/A   | N/A    |
| 31 | Tensor<[28, 28]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                           | Done     | Unknown    | N/A   | N/A    |
| 32 | Tensor<[3, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 33 | Tensor<[3, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 34 | Tensor<[3, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 35 | Tensor<[3, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 36 | Tensor<[3, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 37 | Tensor<[3, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 38 | Tensor<[3, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 39 | Tensor<[3, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 40 | Tensor<[3, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 41 | Tensor<[4, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 42 | Tensor<[4, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 43 | Tensor<[4, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 44 | Tensor<[4, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 45 | Tensor<[4, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 46 | Tensor<[4, 28]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 47 | Tensor<[4, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 48 | Tensor<[4, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 49 | Tensor<[4, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Done     | Unknown    | N/A   | N/A    |
| 50 | Tensor<[49, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 51 | Tensor<[49, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                          | Done     | Unknown    | N/A   | N/A    |
| 52 | Tensor<[49, 56]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                           | Removed  | Unknown    | N/A   | N/A    |
| 53 | Tensor<[56, 56]> self = ?,<br>int dim = 0,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                         | Done     | Unknown    | N/A   | N/A    |
| 54 | Tensor<[56, 56]> self = ?,<br>int dim = 0,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                          | Done     | Unknown    | N/A   | N/A    |
| 55 | Tensor<[56, 56]> self = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                           | Done     | Unknown    | N/A   | N/A    |
| 56 | Tensor<[7, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807                          | Done     | Unknown    | N/A   | N/A    |
| 57 | Tensor<[7, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                                           | Done     | Unknown    | N/A   | N/A    |
| 58 | Tensor<[7, 14]> self = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                                            | Removed  | Unknown    | N/A   | N/A    |
### aten.slice_scatter.default
|    | ATen Input Variations                                                                                                                          | Status   | Isolated   | PCC   | Host   |
|---:|:-----------------------------------------------------------------------------------------------------------------------------------------------|:---------|:-----------|:------|:-------|
|  0 | Tensor<[14, 14]> self = ?,<br>Tensor<[3, 14]> src = ?,<br>int dim = 0,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807 | Done     | Unknown    | N/A   | N/A    |
|  1 | Tensor<[14, 14]> self = ?,<br>Tensor<[4, 14]> src = ?,<br>int dim = 0,<br>Optional[int] start = -7,<br>Optional[int] end = -3                  | Done     | Unknown    | N/A   | N/A    |
|  2 | Tensor<[14, 14]> self = ?,<br>Tensor<[7, 14]> src = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -7                   | Done     | Unknown    | N/A   | N/A    |
|  3 | Tensor<[21, 28]> self = ?,<br>Tensor<[21, 21]> src = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                  | Done     | Unknown    | N/A   | N/A    |
|  4 | Tensor<[21, 28]> self = ?,<br>Tensor<[21, 3]> src = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807 | Done     | Unknown    | N/A   | N/A    |
|  5 | Tensor<[21, 28]> self = ?,<br>Tensor<[21, 4]> src = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                  | Done     | Unknown    | N/A   | N/A    |
|  6 | Tensor<[28, 28]> self = ?,<br>Tensor<[21, 28]> src = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -7                  | Done     | Unknown    | N/A   | N/A    |
|  7 | Tensor<[28, 28]> self = ?,<br>Tensor<[3, 28]> src = ?,<br>int dim = 0,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807 | Done     | Unknown    | N/A   | N/A    |
|  8 | Tensor<[28, 28]> self = ?,<br>Tensor<[4, 28]> src = ?,<br>int dim = 0,<br>Optional[int] start = -7,<br>Optional[int] end = -3                  | Done     | Unknown    | N/A   | N/A    |
|  9 | Tensor<[3, 14]> self = ?,<br>Tensor<[3, 3]> src = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807   | Done     | Unknown    | N/A   | N/A    |
| 10 | Tensor<[3, 14]> self = ?,<br>Tensor<[3, 4]> src = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                    | Done     | Unknown    | N/A   | N/A    |
| 11 | Tensor<[3, 14]> self = ?,<br>Tensor<[3, 7]> src = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                     | Done     | Unknown    | N/A   | N/A    |
| 12 | Tensor<[3, 28]> self = ?,<br>Tensor<[3, 21]> src = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                    | Done     | Unknown    | N/A   | N/A    |
| 13 | Tensor<[3, 28]> self = ?,<br>Tensor<[3, 3]> src = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807   | Done     | Unknown    | N/A   | N/A    |
| 14 | Tensor<[3, 28]> self = ?,<br>Tensor<[3, 4]> src = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                    | Done     | Unknown    | N/A   | N/A    |
| 15 | Tensor<[3, 56]> self = ?,<br>Tensor<[3, 3]> src = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807   | Done     | Unknown    | N/A   | N/A    |
| 16 | Tensor<[3, 56]> self = ?,<br>Tensor<[3, 49]> src = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                    | Done     | Unknown    | N/A   | N/A    |
| 17 | Tensor<[3, 56]> self = ?,<br>Tensor<[3, 4]> src = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                    | Done     | Unknown    | N/A   | N/A    |
| 18 | Tensor<[4, 14]> self = ?,<br>Tensor<[4, 3]> src = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807   | Done     | Unknown    | N/A   | N/A    |
| 19 | Tensor<[4, 14]> self = ?,<br>Tensor<[4, 4]> src = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                    | Done     | Unknown    | N/A   | N/A    |
| 20 | Tensor<[4, 14]> self = ?,<br>Tensor<[4, 7]> src = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                     | Done     | Unknown    | N/A   | N/A    |
| 21 | Tensor<[4, 28]> self = ?,<br>Tensor<[4, 21]> src = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                    | Done     | Unknown    | N/A   | N/A    |
| 22 | Tensor<[4, 28]> self = ?,<br>Tensor<[4, 3]> src = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807   | Done     | Unknown    | N/A   | N/A    |
| 23 | Tensor<[4, 28]> self = ?,<br>Tensor<[4, 4]> src = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                    | Done     | Unknown    | N/A   | N/A    |
| 24 | Tensor<[4, 56]> self = ?,<br>Tensor<[4, 3]> src = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807   | Done     | Unknown    | N/A   | N/A    |
| 25 | Tensor<[4, 56]> self = ?,<br>Tensor<[4, 49]> src = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                    | Done     | Unknown    | N/A   | N/A    |
| 26 | Tensor<[4, 56]> self = ?,<br>Tensor<[4, 4]> src = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                    | Done     | Unknown    | N/A   | N/A    |
| 27 | Tensor<[49, 56]> self = ?,<br>Tensor<[49, 3]> src = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807 | Done     | Unknown    | N/A   | N/A    |
| 28 | Tensor<[49, 56]> self = ?,<br>Tensor<[49, 49]> src = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                  | Done     | Unknown    | N/A   | N/A    |
| 29 | Tensor<[49, 56]> self = ?,<br>Tensor<[49, 4]> src = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                  | Done     | Unknown    | N/A   | N/A    |
| 30 | Tensor<[56, 56]> self = ?,<br>Tensor<[3, 56]> src = ?,<br>int dim = 0,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807 | Done     | Unknown    | N/A   | N/A    |
| 31 | Tensor<[56, 56]> self = ?,<br>Tensor<[4, 56]> src = ?,<br>int dim = 0,<br>Optional[int] start = -7,<br>Optional[int] end = -3                  | Done     | Unknown    | N/A   | N/A    |
| 32 | Tensor<[56, 56]> self = ?,<br>Tensor<[49, 56]> src = ?,<br>int dim = 0,<br>Optional[int] start = 0,<br>Optional[int] end = -7                  | Done     | Unknown    | N/A   | N/A    |
| 33 | Tensor<[7, 14]> self = ?,<br>Tensor<[7, 3]> src = ?,<br>int dim = 1,<br>Optional[int] start = -3,<br>Optional[int] end = 9223372036854775807   | Done     | Unknown    | N/A   | N/A    |
| 34 | Tensor<[7, 14]> self = ?,<br>Tensor<[7, 4]> src = ?,<br>int dim = 1,<br>Optional[int] start = -7,<br>Optional[int] end = -3                    | Done     | Unknown    | N/A   | N/A    |
| 35 | Tensor<[7, 14]> self = ?,<br>Tensor<[7, 7]> src = ?,<br>int dim = 1,<br>Optional[int] start = 0,<br>Optional[int] end = -7                     | Done     | Unknown    | N/A   | N/A    |
### aten.sub.Tensor
|    | ATen Input Variations                                          | Status   | Isolated   |      PCC |   Host |
|---:|:---------------------------------------------------------------|:---------|:-----------|---------:|-------:|
|  0 | Tensor<[16, 1, 49]> self = ?,<br>Tensor<[16, 49, 1]> other = ? | Done     | Done       | 0.454187 |      0 |
|  1 | Tensor<[4, 1, 49]> self = ?,<br>Tensor<[4, 49, 1]> other = ?   | Done     | Done       | 0.463294 |      0 |
|  2 | Tensor<[64, 1, 49]> self = ?,<br>Tensor<[64, 49, 1]> other = ? | Done     | Done       | 0.344988 |      0 |
### aten.t.default
|    | ATen Input Variations        | Status   | Isolated   |   PCC |   Host |
|---:|:-----------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1000, 768]> self = ? | Removed  | Done       |     1 |      0 |
|  1 | Tensor<[1152, 384]> self = ? | Removed  | Done       |     1 |      0 |
|  2 | Tensor<[1536, 384]> self = ? | Removed  | Done       |     1 |      0 |
|  3 | Tensor<[192, 192]> self = ?  | Removed  | Done       |     1 |      0 |
|  4 | Tensor<[192, 384]> self = ?  | Removed  | Done       |     1 |      0 |
|  5 | Tensor<[192, 768]> self = ?  | Removed  | Done       |     1 |      0 |
|  6 | Tensor<[2304, 768]> self = ? | Removed  | Done       |     1 |      0 |
|  7 | Tensor<[288, 96]> self = ?   | Removed  | Done       |     1 |      0 |
|  8 | Tensor<[3072, 768]> self = ? | Removed  | Done       |     1 |      0 |
|  9 | Tensor<[384, 1536]> self = ? | Removed  | Done       |     1 |      0 |
| 10 | Tensor<[384, 384]> self = ?  | Removed  | Done       |     1 |      0 |
| 11 | Tensor<[384, 768]> self = ?  | Removed  | Done       |     1 |      0 |
| 12 | Tensor<[384, 96]> self = ?   | Removed  | Done       |     1 |      0 |
| 13 | Tensor<[576, 192]> self = ?  | Removed  | Done       |     1 |      0 |
| 14 | Tensor<[768, 1536]> self = ? | Removed  | Done       |     1 |      0 |
| 15 | Tensor<[768, 192]> self = ?  | Removed  | Done       |     1 |      0 |
| 16 | Tensor<[768, 3072]> self = ? | Removed  | Done       |     1 |      0 |
| 17 | Tensor<[768, 768]> self = ?  | Removed  | Done       |     1 |      0 |
| 18 | Tensor<[96, 384]> self = ?   | Removed  | Done       |     1 |      0 |
| 19 | Tensor<[96, 96]> self = ?    | Removed  | Done       |     1 |      0 |
### aten.transpose.int
|    | ATen Input Variations                                                | Status   | Isolated   |   PCC |   Host |
|---:|:---------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 24, 49, 32]> self = ?,<br>int dim0 = -2,<br>int dim1 = -1 | Done     | Done       |     1 |      0 |
|  1 | Tensor<[1, 24, 49, 32]> self = ?,<br>int dim0 = 1,<br>int dim1 = 2   | Done     | Done       |     1 |      0 |
|  2 | Tensor<[16, 6, 49, 32]> self = ?,<br>int dim0 = -2,<br>int dim1 = -1 | Done     | Done       |     1 |      0 |
|  3 | Tensor<[16, 6, 49, 32]> self = ?,<br>int dim0 = 1,<br>int dim1 = 2   | Done     | Done       |     1 |      0 |
|  4 | Tensor<[4, 12, 49, 32]> self = ?,<br>int dim0 = -2,<br>int dim1 = -1 | Done     | Done       |     1 |      0 |
|  5 | Tensor<[4, 12, 49, 32]> self = ?,<br>int dim0 = 1,<br>int dim1 = 2   | Done     | Done       |     1 |      0 |
|  6 | Tensor<[64, 3, 49, 32]> self = ?,<br>int dim0 = -2,<br>int dim1 = -1 | Done     | Done       |     1 |      0 |
|  7 | Tensor<[64, 3, 49, 32]> self = ?,<br>int dim0 = 1,<br>int dim1 = 2   | Done     | Done       |     1 |      0 |
### aten.unsqueeze.default
|    | ATen Input Variations                            | Status   | Isolated   |   PCC |   Host |
|---:|:-------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[12, 49, 49]> self = ?,<br>int dim = 0    | Removed  | Done       |     1 |      0 |
|  1 | Tensor<[16, 1, 49, 49]> self = ?,<br>int dim = 0 | Done     | Done       |     1 |      0 |
|  2 | Tensor<[16, 49, 49]> self = ?,<br>int dim = 1    | Done     | Done       |     1 |      0 |
|  3 | Tensor<[16, 49]> self = ?,<br>int dim = 1        | Done     | Done       |     1 |      0 |
|  4 | Tensor<[16, 49]> self = ?,<br>int dim = 2        | Done     | Done       |     1 |      0 |
|  5 | Tensor<[24, 49, 49]> self = ?,<br>int dim = 0    | Removed  | Done       |     1 |      0 |
|  6 | Tensor<[3, 49, 49]> self = ?,<br>int dim = 0     | Removed  | Done       |     1 |      0 |
|  7 | Tensor<[4, 1, 49, 49]> self = ?,<br>int dim = 0  | Done     | Done       |     1 |      0 |
|  8 | Tensor<[4, 49, 49]> self = ?,<br>int dim = 1     | Done     | Done       |     1 |      0 |
|  9 | Tensor<[4, 49]> self = ?,<br>int dim = 1         | Done     | Done       |     1 |      0 |
| 10 | Tensor<[4, 49]> self = ?,<br>int dim = 2         | Done     | Done       |     1 |      0 |
| 11 | Tensor<[6, 49, 49]> self = ?,<br>int dim = 0     | Removed  | Done       |     1 |      0 |
| 12 | Tensor<[64, 1, 49, 49]> self = ?,<br>int dim = 0 | Done     | Done       |     1 |      0 |
| 13 | Tensor<[64, 49, 49]> self = ?,<br>int dim = 1    | Done     | Done       |     1 |      0 |
| 14 | Tensor<[64, 49]> self = ?,<br>int dim = 1        | Done     | Done       |     1 |      0 |
| 15 | Tensor<[64, 49]> self = ?,<br>int dim = 2        | Done     | Done       |     1 |      0 |
### aten.view.default
|    | ATen Input Variations                                                       | Status   | Isolated   |   PCC |   Host |
|---:|:----------------------------------------------------------------------------|:---------|:-----------|------:|-------:|
|  0 | Tensor<[1, 1, 1, 7, 7, 768]> self = ?,<br>List[int] size = [1, 49, 768]     | Done     | Done       |     1 |      0 |
|  1 | Tensor<[1, 1, 7, 1, 7, 768]> self = ?,<br>List[int] size = [1, 7, 7, 768]   | Done     | Done       |     1 |      0 |
|  2 | Tensor<[1, 14, 14, 1536]> self = ?,<br>List[int] size = [196, 1536]         | Done     | Done       |     1 |      0 |
|  3 | Tensor<[1, 14, 14, 384]> self = ?,<br>List[int] size = [1, 2, 7, 2, 7, 384] | Done     | Done       |     1 |      0 |
|  4 | Tensor<[1, 14, 14, 384]> self = ?,<br>List[int] size = [196, 384]           | Done     | Done       |     1 |      0 |
|  5 | Tensor<[1, 14, 14, 768]> self = ?,<br>List[int] size = [196, 768]           | Done     | Done       |     1 |      0 |
|  6 | Tensor<[1, 16, 6, 49, 49]> self = ?,<br>List[int] size = [-1, 6, 49, 49]    | Done     | Done       |     1 |      0 |
|  7 | Tensor<[1, 24, 32, 49]> self = ?,<br>List[int] size = [24, 32, 49]          | Done     | Done       |     1 |      0 |
|  8 | Tensor<[1, 24, 49, 32]> self = ?,<br>List[int] size = [24, 49, 32]          | Done     | Done       |     1 |      0 |
|  9 | Tensor<[1, 24, 49, 49]> self = ?,<br>List[int] size = [24, 49, 49]          | Done     | Done       |     1 |      0 |
| 10 | Tensor<[1, 28, 28, 192]> self = ?,<br>List[int] size = [1, 4, 7, 4, 7, 192] | Done     | Done       |     1 |      0 |
| 11 | Tensor<[1, 28, 28, 192]> self = ?,<br>List[int] size = [784, 192]           | Done     | Done       |     1 |      0 |
| 12 | Tensor<[1, 28, 28, 384]> self = ?,<br>List[int] size = [784, 384]           | Done     | Done       |     1 |      0 |
| 13 | Tensor<[1, 28, 28, 768]> self = ?,<br>List[int] size = [784, 768]           | Done     | Done       |     1 |      0 |
| 14 | Tensor<[1, 4, 12, 49, 49]> self = ?,<br>List[int] size = [-1, 12, 49, 49]   | Done     | Done       |     1 |      0 |
| 15 | Tensor<[1, 49, 2304]> self = ?,<br>List[int] size = [1, 49, 3, 24, 32]      | Done     | Done       |     1 |      0 |
| 16 | Tensor<[1, 49, 768]> self = ?,<br>List[int] size = [1, 1, 1, 7, 7, 768]     | Done     | Done       |     1 |      0 |
| 17 | Tensor<[1, 49, 768]> self = ?,<br>List[int] size = [49, 768]                | Done     | Done       |     1 |      0 |
| 18 | Tensor<[1, 56, 56, 384]> self = ?,<br>List[int] size = [3136, 384]          | Done     | Done       |     1 |      0 |
| 19 | Tensor<[1, 56, 56, 96]> self = ?,<br>List[int] size = [1, 8, 7, 8, 7, 96]   | Done     | Done       |     1 |      0 |
| 20 | Tensor<[1, 56, 56, 96]> self = ?,<br>List[int] size = [3136, 96]            | Done     | Done       |     1 |      0 |
| 21 | Tensor<[1, 64, 3, 49, 49]> self = ?,<br>List[int] size = [-1, 3, 49, 49]    | Done     | Done       |     1 |      0 |
| 22 | Tensor<[1, 7, 7, 1536]> self = ?,<br>List[int] size = [49, 1536]            | Done     | Done       |     1 |      0 |
| 23 | Tensor<[1, 7, 7, 3072]> self = ?,<br>List[int] size = [49, 3072]            | Done     | Done       |     1 |      0 |
| 24 | Tensor<[1, 7, 7, 768]> self = ?,<br>List[int] size = [1, 1, 7, 1, 7, 768]   | Done     | Done       |     1 |      0 |
| 25 | Tensor<[1, 7, 7, 768]> self = ?,<br>List[int] size = [49, 768]              | Done     | Done       |     1 |      0 |
| 26 | Tensor<[1, 768, 1, 1]> self = ?,<br>List[int] size = [1, 768]               | Done     | Done       |     1 |      0 |
| 27 | Tensor<[14, 14]> self = ?,<br>List[int] size = [2, 7, 2, 7]                 | Done     | Done       |     1 |      0 |
| 28 | Tensor<[16, 49, 192]> self = ?,<br>List[int] size = [1, 4, 4, 7, 7, 192]    | Done     | Done       |     1 |      0 |
| 29 | Tensor<[16, 49, 192]> self = ?,<br>List[int] size = [784, 192]              | Done     | Done       |     1 |      0 |
| 30 | Tensor<[16, 49, 576]> self = ?,<br>List[int] size = [16, 49, 3, 6, 32]      | Done     | Done       |     1 |      0 |
| 31 | Tensor<[16, 6, 49, 49]> self = ?,<br>List[int] size = [1, 16, 6, 49, 49]    | Done     | Done       |     1 |      0 |
| 32 | Tensor<[16, 6, 49, 49]> self = ?,<br>List[int] size = [96, 49, 49]          | Done     | Done       |     1 |      0 |
| 33 | Tensor<[192, 49, 32]> self = ?,<br>List[int] size = [64, 3, 49, 32]         | Done     | Done       |     1 |      0 |
| 34 | Tensor<[192, 49, 49]> self = ?,<br>List[int] size = [64, 3, 49, 49]         | Done     | Done       |     1 |      0 |
| 35 | Tensor<[196, 1152]> self = ?,<br>List[int] size = [4, 49, 1152]             | Done     | Done       |     1 |      0 |
| 36 | Tensor<[196, 1536]> self = ?,<br>List[int] size = [1, 14, 14, 1536]         | Done     | Done       |     1 |      0 |
| 37 | Tensor<[196, 384]> self = ?,<br>List[int] size = [1, 14, 14, 384]           | Done     | Done       |     1 |      0 |
| 38 | Tensor<[196, 384]> self = ?,<br>List[int] size = [4, 49, 384]               | Done     | Done       |     1 |      0 |
| 39 | Tensor<[24, 49, 32]> self = ?,<br>List[int] size = [1, 24, 49, 32]          | Done     | Done       |     1 |      0 |
| 40 | Tensor<[24, 49, 49]> self = ?,<br>List[int] size = [1, 24, 49, 49]          | Done     | Done       |     1 |      0 |
| 41 | Tensor<[2401, 12]> self = ?,<br>List[int] size = [49, 49, -1]               | Removed  | Done       |     1 |      0 |
| 42 | Tensor<[2401, 24]> self = ?,<br>List[int] size = [49, 49, -1]               | Removed  | Done       |     1 |      0 |
| 43 | Tensor<[2401, 3]> self = ?,<br>List[int] size = [49, 49, -1]                | Removed  | Done       |     1 |      0 |
| 44 | Tensor<[2401, 6]> self = ?,<br>List[int] size = [49, 49, -1]                | Removed  | Done       |     1 |      0 |
| 45 | Tensor<[28, 28]> self = ?,<br>List[int] size = [4, 7, 4, 7]                 | Done     | Done       |     1 |      0 |
| 46 | Tensor<[3136, 288]> self = ?,<br>List[int] size = [64, 49, 288]             | Done     | Done       |     1 |      0 |
| 47 | Tensor<[3136, 384]> self = ?,<br>List[int] size = [1, 56, 56, 384]          | Done     | Done       |     1 |      0 |
| 48 | Tensor<[3136, 96]> self = ?,<br>List[int] size = [1, 56, 56, 96]            | Done     | Done       |     1 |      0 |
| 49 | Tensor<[3136, 96]> self = ?,<br>List[int] size = [64, 49, 96]               | Done     | Done       |     1 |      0 |
| 50 | Tensor<[4, 12, 49, 49]> self = ?,<br>List[int] size = [1, 4, 12, 49, 49]    | Done     | Done       |     1 |      0 |
| 51 | Tensor<[4, 12, 49, 49]> self = ?,<br>List[int] size = [48, 49, 49]          | Done     | Done       |     1 |      0 |
| 52 | Tensor<[4, 49, 1152]> self = ?,<br>List[int] size = [4, 49, 3, 12, 32]      | Done     | Done       |     1 |      0 |
| 53 | Tensor<[4, 49, 384]> self = ?,<br>List[int] size = [1, 2, 2, 7, 7, 384]     | Done     | Done       |     1 |      0 |
| 54 | Tensor<[4, 49, 384]> self = ?,<br>List[int] size = [196, 384]               | Done     | Done       |     1 |      0 |
| 55 | Tensor<[48, 49, 32]> self = ?,<br>List[int] size = [4, 12, 49, 32]          | Done     | Done       |     1 |      0 |
| 56 | Tensor<[48, 49, 49]> self = ?,<br>List[int] size = [4, 12, 49, 49]          | Done     | Done       |     1 |      0 |
| 57 | Tensor<[49, 2304]> self = ?,<br>List[int] size = [1, 49, 2304]              | Done     | Done       |     1 |      0 |
| 58 | Tensor<[49, 3072]> self = ?,<br>List[int] size = [1, 7, 7, 3072]            | Done     | Done       |     1 |      0 |
| 59 | Tensor<[49, 768]> self = ?,<br>List[int] size = [1, 49, 768]                | Done     | Done       |     1 |      0 |
| 60 | Tensor<[49, 768]> self = ?,<br>List[int] size = [1, 7, 7, 768]              | Done     | Done       |     1 |      0 |
| 61 | Tensor<[56, 56]> self = ?,<br>List[int] size = [8, 7, 8, 7]                 | Done     | Done       |     1 |      0 |
| 62 | Tensor<[64, 3, 49, 49]> self = ?,<br>List[int] size = [1, 64, 3, 49, 49]    | Done     | Done       |     1 |      0 |
| 63 | Tensor<[64, 3, 49, 49]> self = ?,<br>List[int] size = [192, 49, 49]         | Done     | Done       |     1 |      0 |
| 64 | Tensor<[64, 49, 288]> self = ?,<br>List[int] size = [64, 49, 3, 3, 32]      | Done     | Done       |     1 |      0 |
| 65 | Tensor<[64, 49, 96]> self = ?,<br>List[int] size = [1, 8, 8, 7, 7, 96]      | Done     | Done       |     1 |      0 |
| 66 | Tensor<[64, 49, 96]> self = ?,<br>List[int] size = [3136, 96]               | Done     | Done       |     1 |      0 |
| 67 | Tensor<[784, 192]> self = ?,<br>List[int] size = [1, 28, 28, 192]           | Done     | Done       |     1 |      0 |
| 68 | Tensor<[784, 192]> self = ?,<br>List[int] size = [16, 49, 192]              | Done     | Done       |     1 |      0 |
| 69 | Tensor<[784, 576]> self = ?,<br>List[int] size = [16, 49, 576]              | Done     | Done       |     1 |      0 |
| 70 | Tensor<[784, 768]> self = ?,<br>List[int] size = [1, 28, 28, 768]           | Done     | Done       |     1 |      0 |
| 71 | Tensor<[96, 49, 32]> self = ?,<br>List[int] size = [16, 6, 49, 32]          | Done     | Done       |     1 |      0 |
| 72 | Tensor<[96, 49, 49]> self = ?,<br>List[int] size = [16, 6, 49, 49]          | Done     | Done       |     1 |      0 |

