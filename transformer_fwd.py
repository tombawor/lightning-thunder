import torch
from nvfuser import FusionDefinition, DataType


def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T1 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]
    )
    T2 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T3 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]
    )
    T4 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T5 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T6 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T7 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T8 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T9 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]
    )
    T10 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T11 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]
    )
    T12 = fd.define_tensor(
        shape=[1, -1, -1], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0]
    )
    T13 = fd.ops.cast(T12, dtype=DataType.Float)
    T14, T15 = fd.ops.var_mean(T13, dims=[2], correction=0, keepdim=False)
    S16 = fd.define_scalar(1, dtype=DataType.Int)
    S17 = fd.define_scalar(2048, dtype=DataType.Int)
    S18 = fd.define_scalar(1, dtype=DataType.Int)
    V19 = fd.define_vector([S16, S17, S18], dtype=DataType.Int)
    T20 = fd.ops.broadcast_in_dim(T14, shape=V19, broadcast_dims=[0, 1])
    S21 = fd.define_scalar(1, dtype=DataType.Int)
    S22 = fd.define_scalar(2048, dtype=DataType.Int)
    S23 = fd.define_scalar(1, dtype=DataType.Int)
    V24 = fd.define_vector([S21, S22, S23], dtype=DataType.Int)
    T25 = fd.ops.broadcast_in_dim(T15, shape=V24, broadcast_dims=[0, 1])
    S26 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T27 = fd.ops.add(T20, S26)
    T28 = fd.ops.rsqrt(T27)
    S29 = fd.define_scalar(1, dtype=DataType.Int)
    S30 = fd.define_scalar(2048, dtype=DataType.Int)
    S31 = fd.define_scalar(12288, dtype=DataType.Int)
    V32 = fd.define_vector([S29, S30, S31], dtype=DataType.Int)
    T33 = fd.ops.broadcast_in_dim(T25, shape=V32, broadcast_dims=[0, 1, 2])
    T34 = fd.ops.sub(T13, T33)
    S35 = fd.define_scalar(1, dtype=DataType.Int)
    S36 = fd.define_scalar(2048, dtype=DataType.Int)
    S37 = fd.define_scalar(12288, dtype=DataType.Int)
    V38 = fd.define_vector([S35, S36, S37], dtype=DataType.Int)
    T39 = fd.ops.broadcast_in_dim(T28, shape=V38, broadcast_dims=[0, 1, 2])
    T40 = fd.ops.mul(T34, T39)
    S41 = fd.define_scalar(1, dtype=DataType.Int)
    S42 = fd.define_scalar(2048, dtype=DataType.Int)
    S43 = fd.define_scalar(12288, dtype=DataType.Int)
    V44 = fd.define_vector([S41, S42, S43], dtype=DataType.Int)
    T45 = fd.ops.broadcast_in_dim(T5, shape=V44, broadcast_dims=[2])
    T46 = fd.ops.cast(T45, dtype=DataType.Float)
    T47 = fd.ops.mul(T40, T46)
    S48 = fd.define_scalar(1, dtype=DataType.Int)
    S49 = fd.define_scalar(2048, dtype=DataType.Int)
    S50 = fd.define_scalar(12288, dtype=DataType.Int)
    V51 = fd.define_vector([S48, S49, S50], dtype=DataType.Int)
    T52 = fd.ops.broadcast_in_dim(T4, shape=V51, broadcast_dims=[2])
    T53 = fd.ops.cast(T52, dtype=DataType.Float)
    T54 = fd.ops.add(T47, T53)
    T55 = fd.ops.cast(T54, dtype=DataType.BFloat16)
    S56 = fd.define_scalar(2048, dtype=DataType.Int)
    S57 = fd.define_scalar(12288, dtype=DataType.Int)
    V58 = fd.define_vector([S56, S57], dtype=DataType.Int)
    T59 = fd.ops.reshape(T55, new_shape=V58)
    T60 = fd.ops.linear(T59, T1, T0)
    S61 = fd.define_scalar(1, dtype=DataType.Int)
    S62 = fd.define_scalar(2048, dtype=DataType.Int)
    S63 = fd.define_scalar(36864, dtype=DataType.Int)
    V64 = fd.define_vector([S61, S62, S63], dtype=DataType.Int)
    T65 = fd.ops.reshape(T60, new_shape=V64)
    T66 = fd.ops.slice(T65, start_indices=[0, 0, 0], end_indices=[1, 2048, 12288], strides=[1, 1, 1])
    T67 = fd.ops.slice(T65, start_indices=[0, 0, 12288], end_indices=[1, 2048, 24576], strides=[1, 1, 1])
    T68 = fd.ops.slice(T65, start_indices=[0, 0, 24576], end_indices=[1, 2048, 36864], strides=[1, 1, 1])
    S69 = fd.define_scalar(1, dtype=DataType.Int)
    S70 = fd.define_scalar(2048, dtype=DataType.Int)
    S71 = fd.define_scalar(96, dtype=DataType.Int)
    S72 = fd.define_scalar(128, dtype=DataType.Int)
    V73 = fd.define_vector([S69, S70, S71, S72], dtype=DataType.Int)
    T74 = fd.ops.reshape(T67, new_shape=V73)
    # The portion between {T82, T75, T89} and {T179} are SDPA compiled from
    # https://github.com/Lightning-AI/lightning-thunder/blob/1aaa463323dfe875ee75da9a3454779b69cb665c/thunder/torch/__init__.py#L4820.
    # T82 = Q, T75 = K, T89 = V, T179 = SDPA output
    T75 = fd.ops.permute(T74, dims=[0, 2, 1, 3])
    S76 = fd.define_scalar(1, dtype=DataType.Int)
    S77 = fd.define_scalar(2048, dtype=DataType.Int)
    S78 = fd.define_scalar(96, dtype=DataType.Int)
    S79 = fd.define_scalar(128, dtype=DataType.Int)
    V80 = fd.define_vector([S76, S77, S78, S79], dtype=DataType.Int)
    T81 = fd.ops.reshape(T66, new_shape=V80)
    T82 = fd.ops.permute(T81, dims=[0, 2, 1, 3])
    S83 = fd.define_scalar(1, dtype=DataType.Int)
    S84 = fd.define_scalar(2048, dtype=DataType.Int)
    S85 = fd.define_scalar(96, dtype=DataType.Int)
    S86 = fd.define_scalar(128, dtype=DataType.Int)
    V87 = fd.define_vector([S83, S84, S85, S86], dtype=DataType.Int)
    T88 = fd.ops.reshape(T68, new_shape=V87)
    T89 = fd.ops.permute(T88, dims=[0, 2, 1, 3])
    T90 = fd.ops.cast(T82, dtype=DataType.Float)
    S91 = fd.define_scalar(0.297302, dtype=DataType.Double)
    T92 = fd.ops.mul(T90, S91)
    T93 = fd.ops.cast(T92, dtype=DataType.BFloat16)
    T94 = fd.ops.permute(T75, dims=[0, 1, 3, 2])
    T95 = fd.ops.cast(T94, dtype=DataType.Float)
    S96 = fd.define_scalar(0.297302, dtype=DataType.Double)
    T97 = fd.ops.mul(T95, S96)
    T98 = fd.ops.cast(T97, dtype=DataType.BFloat16)
    T99 = fd.ops.matmul(T93, T98)
    S100 = fd.define_scalar(2048, dtype=DataType.Int)
    S101 = fd.define_scalar(0, dtype=DataType.Int)
    S102 = fd.define_scalar(1, dtype=DataType.Int)
    T103 = fd.ops.iota(S100, S101, S102, dtype=DataType.Int)
    S104 = fd.define_scalar(2048, dtype=DataType.Int)
    S105 = fd.define_scalar(1, dtype=DataType.Int)
    V106 = fd.define_vector([S104, S105], dtype=DataType.Int)
    T107 = fd.ops.broadcast_in_dim(T103, shape=V106, broadcast_dims=[0])
    S108 = fd.define_scalar(1, dtype=DataType.Int)
    S109 = fd.define_scalar(2048, dtype=DataType.Int)
    V110 = fd.define_vector([S108, S109], dtype=DataType.Int)
    T111 = fd.ops.broadcast_in_dim(T103, shape=V110, broadcast_dims=[1])
    S112 = fd.define_scalar(0, dtype=DataType.Int)
    T113 = fd.ops.add(T107, S112)
    S114 = fd.define_scalar(2048, dtype=DataType.Int)
    S115 = fd.define_scalar(2048, dtype=DataType.Int)
    V116 = fd.define_vector([S114, S115], dtype=DataType.Int)
    T117 = fd.ops.broadcast_in_dim(T113, shape=V116, broadcast_dims=[0, 1])
    S118 = fd.define_scalar(2048, dtype=DataType.Int)
    S119 = fd.define_scalar(2048, dtype=DataType.Int)
    V120 = fd.define_vector([S118, S119], dtype=DataType.Int)
    T121 = fd.ops.broadcast_in_dim(T111, shape=V120, broadcast_dims=[0, 1])
    T122 = fd.ops.ge(T117, T121)
    S123 = fd.define_scalar(1, dtype=DataType.Int)
    S124 = fd.define_scalar(96, dtype=DataType.Int)
    S125 = fd.define_scalar(2048, dtype=DataType.Int)
    S126 = fd.define_scalar(2048, dtype=DataType.Int)
    V127 = fd.define_vector([S123, S124, S125, S126], dtype=DataType.Int)
    T128 = fd.ops.broadcast_in_dim(T122, shape=V127, broadcast_dims=[2, 3])
    S129 = fd.define_scalar(float("-inf"), dtype=DataType.Double)
    T130 = fd.ops.where(T128, T99, S129)
    T131 = fd.ops.cast(T130, dtype=DataType.Float)
    T132 = fd.ops.max(T131, dims=[3], keepdim=False, dtype=DataType.Null)
    S133 = fd.define_scalar(1, dtype=DataType.Int)
    S134 = fd.define_scalar(96, dtype=DataType.Int)
    S135 = fd.define_scalar(2048, dtype=DataType.Int)
    S136 = fd.define_scalar(1, dtype=DataType.Int)
    V137 = fd.define_vector([S133, S134, S135, S136], dtype=DataType.Int)
    T138 = fd.ops.broadcast_in_dim(T132, shape=V137, broadcast_dims=[0, 1, 2])
    S139 = fd.define_scalar(1, dtype=DataType.Int)
    S140 = fd.define_scalar(96, dtype=DataType.Int)
    S141 = fd.define_scalar(2048, dtype=DataType.Int)
    S142 = fd.define_scalar(2048, dtype=DataType.Int)
    V143 = fd.define_vector([S139, S140, S141, S142], dtype=DataType.Int)
    T144 = fd.ops.broadcast_in_dim(T138, shape=V143, broadcast_dims=[0, 1, 2, 3])
    T145 = fd.ops.sub(T131, T144)
    T146 = fd.ops.exp(T145)
    T147 = fd.ops.sum(T146, dims=[3], keepdim=False, dtype=DataType.Null)
    S148 = fd.define_scalar(1, dtype=DataType.Int)
    S149 = fd.define_scalar(96, dtype=DataType.Int)
    S150 = fd.define_scalar(2048, dtype=DataType.Int)
    S151 = fd.define_scalar(1, dtype=DataType.Int)
    V152 = fd.define_vector([S148, S149, S150, S151], dtype=DataType.Int)
    T153 = fd.ops.broadcast_in_dim(T147, shape=V152, broadcast_dims=[0, 1, 2])
    S154 = fd.define_scalar(1, dtype=DataType.Int)
    S155 = fd.define_scalar(96, dtype=DataType.Int)
    S156 = fd.define_scalar(2048, dtype=DataType.Int)
    S157 = fd.define_scalar(2048, dtype=DataType.Int)
    V158 = fd.define_vector([S154, S155, S156, S157], dtype=DataType.Int)
    T159 = fd.ops.broadcast_in_dim(T153, shape=V158, broadcast_dims=[0, 1, 2, 3])
    T160 = fd.ops.reciprocal(T159)
    T161 = fd.ops.mul(T146, T160)
    T162 = fd.ops.cast(T161, dtype=DataType.BFloat16)
    S163 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S164 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S165 = fd.define_scalar(1, dtype=DataType.Int)
    S166 = fd.define_scalar(96, dtype=DataType.Int)
    S167 = fd.define_scalar(2048, dtype=DataType.Int)
    S168 = fd.define_scalar(2048, dtype=DataType.Int)
    V169 = fd.define_vector([S165, S166, S167, S168], dtype=DataType.Int)
    T170 = fd.ops.uniform(S163, S164, shape=V169, dtype=DataType.BFloat16)
    S171 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T172 = fd.ops.lt(T170, S171)
    T173 = fd.ops.cast(T172, dtype=DataType.Float)
    T174 = fd.ops.mul(T161, T173)
    S175 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T176 = fd.ops.mul(T174, S175)
    T177 = fd.ops.cast(T176, dtype=DataType.BFloat16)
    T178 = fd.ops.matmul(T177, T89)
    T179 = fd.ops.permute(T178, dims=[0, 2, 1, 3])
    T180 = fd.ops.stride_order(T179, stride_order=[3, 2, 1, 0])
    S181 = fd.define_scalar(1, dtype=DataType.Int)
    S182 = fd.define_scalar(2048, dtype=DataType.Int)
    S183 = fd.define_scalar(12288, dtype=DataType.Int)
    V184 = fd.define_vector([S181, S182, S183], dtype=DataType.Int)
    T185 = fd.ops.reshape(T180, new_shape=V184)
    S186 = fd.define_scalar(2048, dtype=DataType.Int)
    S187 = fd.define_scalar(12288, dtype=DataType.Int)
    V188 = fd.define_vector([S186, S187], dtype=DataType.Int)
    T189 = fd.ops.reshape(T185, new_shape=V188)
    T190 = fd.ops.linear(T189, T3, T2)
    S191 = fd.define_scalar(1, dtype=DataType.Int)
    S192 = fd.define_scalar(2048, dtype=DataType.Int)
    S193 = fd.define_scalar(12288, dtype=DataType.Int)
    V194 = fd.define_vector([S191, S192, S193], dtype=DataType.Int)
    T195 = fd.ops.reshape(T190, new_shape=V194)
    S196 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S197 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S198 = fd.define_scalar(1, dtype=DataType.Int)
    S199 = fd.define_scalar(2048, dtype=DataType.Int)
    S200 = fd.define_scalar(12288, dtype=DataType.Int)
    V201 = fd.define_vector([S198, S199, S200], dtype=DataType.Int)
    T202 = fd.ops.uniform(S196, S197, shape=V201, dtype=DataType.BFloat16)
    S203 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T204 = fd.ops.lt(T202, S203)
    T205 = fd.ops.cast(T195, dtype=DataType.Float)
    T206 = fd.ops.cast(T204, dtype=DataType.Float)
    T207 = fd.ops.mul(T205, T206)
    S208 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T209 = fd.ops.mul(T207, S208)
    T210 = fd.ops.add(T13, T209)
    T211, T212 = fd.ops.var_mean(T210, dims=[2], correction=0, keepdim=False)
    S213 = fd.define_scalar(1, dtype=DataType.Int)
    S214 = fd.define_scalar(2048, dtype=DataType.Int)
    S215 = fd.define_scalar(1, dtype=DataType.Int)
    V216 = fd.define_vector([S213, S214, S215], dtype=DataType.Int)
    T217 = fd.ops.broadcast_in_dim(T211, shape=V216, broadcast_dims=[0, 1])
    S218 = fd.define_scalar(1, dtype=DataType.Int)
    S219 = fd.define_scalar(2048, dtype=DataType.Int)
    S220 = fd.define_scalar(1, dtype=DataType.Int)
    V221 = fd.define_vector([S218, S219, S220], dtype=DataType.Int)
    T222 = fd.ops.broadcast_in_dim(T212, shape=V221, broadcast_dims=[0, 1])
    S223 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T224 = fd.ops.add(T217, S223)
    T225 = fd.ops.rsqrt(T224)
    S226 = fd.define_scalar(1, dtype=DataType.Int)
    S227 = fd.define_scalar(2048, dtype=DataType.Int)
    S228 = fd.define_scalar(12288, dtype=DataType.Int)
    V229 = fd.define_vector([S226, S227, S228], dtype=DataType.Int)
    T230 = fd.ops.broadcast_in_dim(T222, shape=V229, broadcast_dims=[0, 1, 2])
    T231 = fd.ops.sub(T210, T230)
    S232 = fd.define_scalar(1, dtype=DataType.Int)
    S233 = fd.define_scalar(2048, dtype=DataType.Int)
    S234 = fd.define_scalar(12288, dtype=DataType.Int)
    V235 = fd.define_vector([S232, S233, S234], dtype=DataType.Int)
    T236 = fd.ops.broadcast_in_dim(T225, shape=V235, broadcast_dims=[0, 1, 2])
    T237 = fd.ops.mul(T231, T236)
    S238 = fd.define_scalar(1, dtype=DataType.Int)
    S239 = fd.define_scalar(2048, dtype=DataType.Int)
    S240 = fd.define_scalar(12288, dtype=DataType.Int)
    V241 = fd.define_vector([S238, S239, S240], dtype=DataType.Int)
    T242 = fd.ops.broadcast_in_dim(T7, shape=V241, broadcast_dims=[2])
    T243 = fd.ops.cast(T242, dtype=DataType.Float)
    T244 = fd.ops.mul(T237, T243)
    S245 = fd.define_scalar(1, dtype=DataType.Int)
    S246 = fd.define_scalar(2048, dtype=DataType.Int)
    S247 = fd.define_scalar(12288, dtype=DataType.Int)
    V248 = fd.define_vector([S245, S246, S247], dtype=DataType.Int)
    T249 = fd.ops.broadcast_in_dim(T6, shape=V248, broadcast_dims=[2])
    T250 = fd.ops.cast(T249, dtype=DataType.Float)
    T251 = fd.ops.add(T244, T250)
    T252 = fd.ops.cast(T251, dtype=DataType.BFloat16)
    S253 = fd.define_scalar(2048, dtype=DataType.Int)
    S254 = fd.define_scalar(12288, dtype=DataType.Int)
    V255 = fd.define_vector([S253, S254], dtype=DataType.Int)
    T256 = fd.ops.reshape(T252, new_shape=V255)
    T257 = fd.ops.linear(T256, T9, T8)
    S258 = fd.define_scalar(1, dtype=DataType.Int)
    S259 = fd.define_scalar(2048, dtype=DataType.Int)
    S260 = fd.define_scalar(49152, dtype=DataType.Int)
    V261 = fd.define_vector([S258, S259, S260], dtype=DataType.Int)
    T262 = fd.ops.reshape(T257, new_shape=V261)
    T263 = fd.ops.cast(T262, dtype=DataType.Float)
    T264 = fd.ops.mul(T263, T263)
    T265 = fd.ops.mul(T264, T263)
    S266 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T267 = fd.ops.mul(S266, T263)
    S268 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T269 = fd.ops.mul(S268, T265)
    T270 = fd.ops.add(T263, T269)
    S271 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T272 = fd.ops.mul(S271, T270)
    T273 = fd.ops.tanh(T272)
    S274 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T275 = fd.ops.add(S274, T273)
    T276 = fd.ops.mul(T267, T275)
    T277 = fd.ops.cast(T276, dtype=DataType.BFloat16)
    S278 = fd.define_scalar(2048, dtype=DataType.Int)
    S279 = fd.define_scalar(49152, dtype=DataType.Int)
    V280 = fd.define_vector([S278, S279], dtype=DataType.Int)
    T281 = fd.ops.reshape(T277, new_shape=V280)
    T282 = fd.ops.linear(T281, T11, T10)
    S283 = fd.define_scalar(1, dtype=DataType.Int)
    S284 = fd.define_scalar(2048, dtype=DataType.Int)
    S285 = fd.define_scalar(12288, dtype=DataType.Int)
    V286 = fd.define_vector([S283, S284, S285], dtype=DataType.Int)
    T287 = fd.ops.reshape(T282, new_shape=V286)
    S288 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S289 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S290 = fd.define_scalar(1, dtype=DataType.Int)
    S291 = fd.define_scalar(2048, dtype=DataType.Int)
    S292 = fd.define_scalar(12288, dtype=DataType.Int)
    V293 = fd.define_vector([S290, S291, S292], dtype=DataType.Int)
    T294 = fd.ops.uniform(S288, S289, shape=V293, dtype=DataType.BFloat16)
    S295 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T296 = fd.ops.lt(T294, S295)
    T297 = fd.ops.cast(T287, dtype=DataType.Float)
    T298 = fd.ops.cast(T296, dtype=DataType.Float)
    T299 = fd.ops.mul(T297, T298)
    S300 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T301 = fd.ops.mul(T299, S300)
    T302 = fd.ops.add(T210, T301)
    T303 = fd.ops.cast(T302, dtype=DataType.BFloat16)
    fd.add_output(T210)
    fd.add_output(T212)
    fd.add_output(T225)
    fd.add_output(T296)
    fd.add_output(T303)
    fd.add_output(T15)
    fd.add_output(T89)
    fd.add_output(T162)
    fd.add_output(T172)
    fd.add_output(T28)
    fd.add_output(T177)
    fd.add_output(T204)


with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    torch.randn((36864,), dtype=torch.bfloat16, device="cuda:0").as_strided((36864,), (1,)),
    torch.randn((452984832,), dtype=torch.bfloat16, device="cuda:0").as_strided((36864, 12288), (12288, 1)),
    torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided((12288,), (1,)),
    torch.randn((150994944,), dtype=torch.bfloat16, device="cuda:0").as_strided((12288, 12288), (12288, 1)),
    torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided((12288,), (1,)),
    torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided((12288,), (1,)),
    torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided((12288,), (1,)),
    torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided((12288,), (1,)),
    torch.randn((49152,), dtype=torch.bfloat16, device="cuda:0").as_strided((49152,), (1,)),
    torch.randn((603979776,), dtype=torch.bfloat16, device="cuda:0").as_strided((49152, 12288), (12288, 1)),
    torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided((12288,), (1,)),
    torch.randn((603979776,), dtype=torch.bfloat16, device="cuda:0").as_strided((12288, 49152), (49152, 1)),
    torch.randn((25165824,), dtype=torch.bfloat16, device="cuda:0").as_strided((1, 2048, 12288), (25165824, 12288, 1)),
]
fd.execute(inputs)
