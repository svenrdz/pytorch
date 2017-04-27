from torch.autograd.function import Function
from torch._thnn import type2backend
from torch import arange, cat, FloatTensor

from . import _all_functions
from torch.nn.modules.utils import _single, _pair, _triple


class MaxPool1d(Function):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.pad = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        if (input.dim() != 3):
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

        input2d = input.unsqueeze(2)    # size = N*C*1*L
        backend = type2backend[type(input)]
        indices, output = input2d.new().long(), input2d.new()
        backend.SpatialDilatedMaxPooling_updateOutput(backend.library_state,
                                                      input2d, output, indices,
                                                      self.kernel_size, 1,
                                                      self.stride, 1,
                                                      self.pad, 0,
                                                      self.dilation, 1,
                                                      self.ceil_mode)
        indices = indices.squeeze(2)
        output = output.squeeze(2)
        if self.return_indices:
            self.save_for_backward(input, indices)
            self.mark_non_differentiable(indices)
            return output, indices
        else:
            self.save_for_backward(input)
            self.indices = indices
            return output

    def backward(self, grad_output, _indices_grad=None):
        if self.return_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices

        input2d = input.unsqueeze(2)
        indices2d = indices.unsqueeze(2)
        grad_output2d = grad_output.unsqueeze(2)
        grad_input = grad_output2d.new()
        backend = type2backend[type(input)]
        backend.SpatialDilatedMaxPooling_updateGradInput(backend.library_state,
                                                         input2d, grad_output2d, grad_input, indices2d,
                                                         self.kernel_size, 1,
                                                         self.stride, 1,
                                                         self.pad, 0,
                                                         self.dilation, 1,
                                                         self.ceil_mode)
        grad_input = grad_input.squeeze(2)
        return grad_input


class MaxPool2d(Function):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.SpatialDilatedMaxPooling_updateOutput(backend.library_state,
                                                      input, output, indices,
                                                      self.kernel_size[1], self.kernel_size[0],
                                                      self.stride[1], self.stride[0],
                                                      self.padding[1], self.padding[0],
                                                      self.dilation[1], self.dilation[0],
                                                      self.ceil_mode)
        if self.return_indices:
            self.save_for_backward(input, indices)
            self.mark_non_differentiable(indices)
            return output, indices
        else:
            self.save_for_backward(input)
            self.indices = indices
            return output

    def backward(self, grad_output, _indices_grad=None):
        if self.return_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialDilatedMaxPooling_updateGradInput(backend.library_state,
                                                         input, grad_output, grad_input, indices,
                                                         self.kernel_size[1], self.kernel_size[0],
                                                         self.stride[1], self.stride[0],
                                                         self.padding[1], self.padding[0],
                                                         self.dilation[1], self.dilation[0],
                                                         self.ceil_mode)
        return grad_input


class MaxPool3d(Function):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if stride is not None else kernel_size)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.VolumetricDilatedMaxPooling_updateOutput(backend.library_state,
                                                         input, output, indices,
                                                         self.kernel_size[0], self.kernel_size[2], self.kernel_size[1],
                                                         self.stride[0], self.stride[2], self.stride[1],
                                                         self.padding[0], self.padding[2], self.padding[1],
                                                         self.dilation[0], self.dilation[2], self.dilation[1],
                                                         self.ceil_mode)
        if self.return_indices:
            self.save_for_backward(input, indices)
            self.mark_non_differentiable(indices)
            return output, indices
        else:
            self.save_for_backward(input)
            self.indices = indices
            return output

    def backward(self, grad_output, _indices_grad=None):
        if self.return_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.VolumetricDilatedMaxPooling_updateGradInput(backend.library_state,
                                                            input, grad_output, grad_input, indices,
                                                            self.kernel_size[0], self.kernel_size[
                                                                2], self.kernel_size[1],
                                                            self.stride[0], self.stride[2], self.stride[1],
                                                            self.padding[0], self.padding[2], self.padding[1],
                                                            self.dilation[0], self.dilation[2], self.dilation[1],
                                                            self.ceil_mode)
        return grad_input


class MaxUnpool2d(Function):

    def __init__(self, output_size):
        super(MaxUnpool2d, self).__init__()
        self.output_size = output_size

    def forward(self, input, indices):
        self.save_for_backward(input, indices)
        self._backend = type2backend[type(input)]
        output = input.new()
        self._backend.SpatialMaxUnpooling_updateOutput(
            self._backend.library_state, input, output, indices,
            self.output_size[1], self.output_size[0])
        return output

    def backward(self, grad_output):
        input, indices = self.saved_tensors
        grad_input = grad_output.new()
        self._backend.SpatialMaxUnpooling_updateGradInput(
            self._backend.library_state, input, grad_output, grad_input,
            indices, self.output_size[1], self.output_size[0])
        return grad_input, None


class MaxUnpool3d(Function):

    def __init__(self, output_size, stride, padding):
        super(MaxUnpool3d, self).__init__()
        self.output_size = output_size
        self.stride = stride
        self.padding = padding

    def forward(self, input, indices):
        self.save_for_backward(input, indices)
        self._backend = type2backend[type(input)]
        output = input.new()
        self._backend.VolumetricMaxUnpooling_updateOutput(
            self._backend.library_state, input, output, indices,
            self.output_size[0], self.output_size[2], self.output_size[1],
            self.stride[0], self.stride[2], self.stride[1],
            self.padding[0], self.padding[2], self.padding[1])
        return output

    def backward(self, grad_output):
        input, indices = self.saved_tensors
        grad_input = grad_output.new()
        self._backend.VolumetricMaxUnpooling_updateGradInput(
            self._backend.library_state, input, grad_output, grad_input, indices,
            self.output_size[0], self.output_size[2], self.output_size[1],
            self.stride[0], self.stride[2], self.stride[1],
            self.padding[0], self.padding[2], self.padding[1])
        return grad_input, None


class FractionalMaxPool2d(Function):

    def __init__(self, kh, kw, output_size=None, output_ratio=None,
                 return_indices=False, _random_samples=None):
        super(FractionalMaxPool2d, self).__init__()

        # Pool size (how wide the pooling for each output unit is)
        self.kw, self.kh = kw, kh

        # Random samples are drawn for all
        # batch * plane * (height, width; i.e., 2) points. This determines
        # the 2d "pseudorandom" overlapping pooling regions for each
        # (batch element x input plane).
        self.random_samples = _random_samples

        self.return_indices = return_indices

        if output_size is not None:
            self.oh, self.ow = output_size
            self.rh, self.rw = None, None
        elif output_ratio is not None:
            self.oh, self.ow = None, None
            self.rh, self.rw = output_ratio
            assert 0 < self.rh < 1
            assert 0 < self.rw < 1
        else:
            assert False

    def forward(self, input):
        if self.random_samples is None:
            random_samples = input.new().resize_(input.size(0),
                                                 input.size(1), 2).uniform_()
        else:
            random_samples = self.random_samples
            self.random_samples = None

        if self.oh is None:
            self.oh = int(input.size(2) * self.rh)
            self.ow = int(input.size(3) * self.rw)
        assert isinstance(self.oh, int) and isinstance(self.ow, int)

        indices = input.new().long()
        output = input.new()
        self._backend = type2backend[type(input)]
        self._backend.SpatialFractionalMaxPooling_updateOutput(
            self._backend.library_state,
            input,
            output,
            self.ow, self.oh,
            self.kw, self.kh,
            indices,
            random_samples
        )

        self.random_samples = None  # Free unnecessary buffers
        if self.return_indices:
            self.save_for_backward(input, indices)
            return output, indices
        else:
            self.indices = indices
            self.save_for_backward(input)
            return output

    def backward(self, grad_output, _grad_indices=None):
        if self.return_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices

        grad_input = grad_output.new()
        self._backend.SpatialFractionalMaxPooling_updateGradInput(
            self._backend.library_state,
            input,
            grad_output,
            grad_input,
            self.ow, self.oh,
            self.kw, self.kh,
            indices)

        return grad_input


class AvgPool2d(Function):

    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True):
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        backend = type2backend[type(input)]
        output = input.new()
        # can avoid this with cudnn
        self.save_for_backward(input)
        backend.SpatialAveragePooling_updateOutput(
            backend.library_state,
            input, output,
            self.kernel_size[1], self.kernel_size[0],
            self.stride[1], self.stride[0],
            self.padding[1], self.padding[0],
            self.ceil_mode, self.count_include_pad)
        return output

    def backward(self, grad_output):
        backend = type2backend[type(grad_output)]
        input, = self.saved_tensors
        grad_input = grad_output.new()
        backend.SpatialAveragePooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input,
            self.kernel_size[1], self.kernel_size[0],
            self.stride[1], self.stride[0],
            self.padding[1], self.padding[0],
            self.ceil_mode, self.count_include_pad)
        return grad_input


class AvgPool3d(Function):

    def __init__(self, kernel_size, stride=None):
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if stride is not None else kernel_size)

    def forward(self, input):
        backend = type2backend[type(input)]
        output = input.new()
        # can avoid this with cudnn
        self.save_for_backward(input)
        backend.VolumetricAveragePooling_updateOutput(backend.library_state,
                                                      input, output,
                                                      self.kernel_size[0], self.kernel_size[2], self.kernel_size[1],
                                                      self.stride[0], self.stride[2], self.stride[1])
        return output

    def backward(self, grad_output):
        backend = type2backend[type(grad_output)]
        input, = self.saved_tensors
        grad_input = grad_output.new()
        backend.VolumetricAveragePooling_updateGradInput(backend.library_state,
                                                         input, grad_output, grad_input,
                                                         self.kernel_size[0], self.kernel_size[2], self.kernel_size[1],
                                                         self.stride[0], self.stride[2], self.stride[1])
        return grad_input


class AdaptiveMaxPool1d(Function):

    def __init__(self, output_size, return_indices=False):
        self.output_size = _single(output_size)
        self.return_indices = return_indices

    def forward(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

        input2d = input.unsqueeze(2)    # size = N*C*1*L
        backend = type2backend[type(input)]
        indices, output = input2d.new().long(), input2d.new()
        backend.SpatialAdaptiveMaxPooling_updateOutput(backend.library_state,
                                                       input2d, output, indices,
                                                       self.output_size[0], 1)
        indices = indices.squeeze(2)
        output = output.squeeze(2)
        if self.return_indices:
            self.save_for_backward(input, indices)
            self.mark_non_differentiable(indices)
            return output, indices
        else:
            self.save_for_backward(input)
            self.indices = indices
            return output

    def backward(self, grad_output, _indices_grad=None):
        if self.return_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices

        input2d = input.unsqueeze(2)
        indices2d = indices.unsqueeze(2)
        grad_output2d = grad_output.unsqueeze(2)
        grad_input = grad_output2d.new()
        backend = type2backend[type(input)]
        backend.SpatialAdaptiveMaxPooling_updateGradInput(backend.library_state,
                                                          input2d, grad_output2d, grad_input, indices2d)
        grad_input = grad_input.squeeze(2)
        return grad_input


class AdaptiveMaxPool2d(Function):

    def __init__(self, output_size, return_indices=False):
        self.output_size = _pair(output_size)
        self.return_indices = return_indices

    def forward(self, input):
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.SpatialAdaptiveMaxPooling_updateOutput(backend.library_state,
                                                       input, output, indices,
                                                       self.output_size[1], self.output_size[0])
        if self.return_indices:
            self.save_for_backward(input, indices)
            self.mark_non_differentiable(indices)
            return output, indices
        else:
            self.save_for_backward(input)
            self.indices = indices
            return output

    def backward(self, grad_output, _indices_grad=None):
        if self.return_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialAdaptiveMaxPooling_updateGradInput(backend.library_state,
                                                          input, grad_output, grad_input, indices)
        return grad_input


class AdaptiveAvgPool1d(Function):

    def __init__(self, output_size):
        self.output_size = _single(output_size)

    def forward(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

        input2d = input.unsqueeze(2)    # size = N*C*1*L
        backend = type2backend[type(input)]
        output = input2d.new()
        self.save_for_backward(input)
        backend.SpatialAdaptiveAveragePooling_updateOutput(
            backend.library_state,
            input2d, output,
            self.output_size[0], 1)
        output = output.squeeze(2)
        return output

    def backward(self, grad_output):
        backend = type2backend[type(grad_output)]
        input, = self.saved_tensors
        input2d = input.unsqueeze(2)
        grad_output2d = grad_output.unsqueeze(2)
        grad_input = grad_output2d.new()
        backend.SpatialAdaptiveAveragePooling_updateGradInput(
            backend.library_state,
            input2d, grad_output2d, grad_input)
        grad_input = grad_input.squeeze(2)
        return grad_input


class AdaptiveAvgPool2d(Function):

    def __init__(self, output_size):
        self.output_size = _pair(output_size)

    def forward(self, input):
        backend = type2backend[type(input)]
        output = input.new()
        self.save_for_backward(input)
        backend.SpatialAdaptiveAveragePooling_updateOutput(
            backend.library_state,
            input, output,
            self.output_size[1], self.output_size[0])
        return output

    def backward(self, grad_output):
        backend = type2backend[type(grad_output)]
        input, = self.saved_tensors
        grad_input = grad_output.new()
        backend.SpatialAdaptiveAveragePooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input)
        return grad_input


class MAC(Function):

    def forward(self, input):
        H, W = input.size(2), input.size(3)
        wl = min(H, W)
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        self.save_for_backward(input)
        backend.SpatialDilatedMaxPooling_updateOutput(
            backend.library_state,
            input, output, indices,
            wl, wl,  # kernel size
            wl, wl,  # stride
            0, 0,  # padding
            1, 1,  # dilation
            False)
        self.save_for_backward(input)
        self.indices = indices
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        indices = self.indices
        H, W = input.size(2), input.size(3)
        wl = min(H, W)
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialDilatedMaxPooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input, indices,
            wl, wl,  # kernel size
            wl, wl,  # stride
            0, 0,  # padding
            1, 1,  # dilation
            False)
        return grad_input

class RMAC(Function):

    def __init__(self, levels=3, overlap=0.4, eps=1e-4):
        self.levels = levels
        self.overlap = overlap
        self.eps = eps

    def _ratio2regions(self, W, H, w):
        # needs rename (no idea)
        max_steps = max(H, W) // min(H, W)
        overlap = self.overlap
        eps = self.eps
        if H != W:
            steps = arange(0, max_steps).cuda()
            b = steps.add(1).div(max(H, W) - w).pow(-1)
            val = b.mul(w).mul(-1).add(w**2).div(w**2).sub(overlap).abs()
            idx = steps.dot(val.eq(val.min()).float())
            if H < W:
                Wd, Hd = idx, 0
            elif H > W:
                Wd, Hd = 0, idx
        else:
            Wd, Hd = 0, 0
        return Wd, Hd

    def forward(self, input):
        B, K, H, W = input.size()
        w = min(H, W)
        Wd, Hd = self._ratio2regions(H, W, w)
        eps = self.eps

        backend = type2backend[type(input)]
        all_indices = []
        all_output = input.new()
        self.save_for_backward(input)

        for l in range(self.levels):
            output = input.new()
            indices = input.new().long()
            wl = 2 * w // (l + 2)
            Wb = (W - wl) // (l + Wd or 1)
            Hb = (H - wl) // (l + Hd or 1)
            backend.SpatialDilatedMaxPooling_updateOutput(
                backend.library_state,
                input, output, indices,
                wl, wl,  # kernel size
                Wb or wl, Hb or wl,  # stride
                0, 0,  # padding
                1, 1,  # dilation
                False)
            output = output.view(B, K, -1)
            # region_norm = output.norm(2, 1).expand_as(output).add(eps)
            # output = output.div(region_norm).sum(2).squeeze(2)

            all_output = cat([all_output, output], 2)
            all_indices.append(indices)

        region_norms = all_output.norm(2,1).expand_as(all_output).add(eps)
        all_output = all_output.div(region_norms).sum(2).squeeze(2)
        self.all_indices = all_indices

        # Necessary as the output becomes 1 everywhere when dividing by
        # torch.norm(output, 2, 0) if batchsize is one
        if B == 1:
            b_norm = all_output.norm(2) + eps
        else:
            b_norm = all_output.norm(2, 0).expand_as(all_output).add(eps)
        return all_output.div(b_norm)

    def backward(self, all_grad_output):
        input, = self.saved_tensors
        H, W = input.size(2), input.size(3)
        w = min(H, W)
        Wd, Hd = self._ratio2regions(H, W, w)

        all_grad_output = all_grad_output.unsqueeze(2).unsqueeze(2)
        backend = type2backend[type(input)]
        all_indices = self.all_indices
        all_grad_input = all_grad_output.new()

        for l in range(self.levels):
            grad_input = all_grad_output.new()
            indices = all_indices[l]
            grad_output = all_grad_output.expand_as(indices)
            wl = 2 * w // (l + 2)
            Wb = (W - wl) // (l + Wd or 1)
            Hb = (H - wl) // (l + Hd or 1)
            backend.SpatialDilatedMaxPooling_updateGradInput(
                backend.library_state,
                input, grad_output, grad_input, indices,
                wl, wl,  # kernel size
                Wb or wl, Hb or wl,  # stride
                0, 0,  # padding
                1, 1,  # dilation
                False)
            if all_grad_input.dim() == 0:
                all_grad_input = grad_input
            else:
                all_grad_input += grad_input
        return all_grad_input



_all_functions.append(AvgPool2d)
_all_functions.append(AvgPool3d)
_all_functions.append(MaxPool1d)
_all_functions.append(MaxPool2d)
_all_functions.append(MaxPool3d)
_all_functions.append(MaxUnpool2d)
_all_functions.append(MaxUnpool3d)
_all_functions.append(FractionalMaxPool2d)
_all_functions.append(AdaptiveMaxPool1d)
_all_functions.append(AdaptiveMaxPool2d)
_all_functions.append(AdaptiveAvgPool1d)
_all_functions.append(AdaptiveAvgPool2d)
_all_functions.append(MAC)
_all_functions.append(RMAC)
