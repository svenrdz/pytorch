from torch.autograd import Variable
from torch.autograd.function import Function, once_differentiable
from torch._thnn import type2backend
from torch import arange, cat, cuda, numel, FloatTensor
from math import inf

from . import _all_functions
from torch.nn.modules.utils import _single, _pair, _triple


class MaxPool1d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False):
        if (input.dim() != 3):
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))
        ctx.kernel_size = kernel_size
        ctx.stride = stride if stride is not None else kernel_size
        ctx.pad = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode

        input2d = input.unsqueeze(2)    # size = N*C*1*L
        backend = type2backend[type(input)]
        indices, output = input2d.new().long(), input2d.new()
        backend.SpatialDilatedMaxPooling_updateOutput(backend.library_state,
                                                      input2d, output, indices,
                                                      ctx.kernel_size, 1,
                                                      ctx.stride, 1,
                                                      ctx.pad, 0,
                                                      ctx.dilation, 1,
                                                      ctx.ceil_mode)
        indices = indices.squeeze(2)
        output = output.squeeze(2)
        ctx.save_for_backward(input, indices)
        ctx.mark_non_differentiable(indices)
        return output, indices

    @staticmethod
    def backward(ctx, grad_output, _indices_grad=None):
        input, indices = ctx.saved_variables

        grad_input = MaxPool1dBackward.apply(input, indices, grad_output, ctx.kernel_size, ctx.stride, ctx.pad,
                                             ctx.dilation, ctx.ceil_mode)
        return grad_input, None, None, None, None, None, None


class MaxPool1dBackward(Function):

    @staticmethod
    def forward(ctx, input, indices, grad_output, kernel_size, stride, padding, dilation, ceil_mode):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.pad = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        input2d = input.unsqueeze(2)
        indices2d = indices.unsqueeze(2)
        grad_output2d = grad_output.unsqueeze(2)
        grad_input = grad_output2d.new()
        ctx.save_for_backward(indices)
        backend = type2backend[type(input)]
        backend.SpatialDilatedMaxPooling_updateGradInput(backend.library_state,
                                                         input2d, grad_output2d, grad_input, indices2d,
                                                         ctx.kernel_size, 1,
                                                         ctx.stride, 1,
                                                         ctx.pad, 0,
                                                         ctx.dilation, 1,
                                                         ctx.ceil_mode)
        grad_input = grad_input.squeeze(2)
        return grad_input

    @staticmethod
    def backward(ctx, ggI, ggIndices=None):
        indices, = ctx.saved_variables
        gI = Variable(ggI.data.new(ggI.size()).zero_())
        ggO = ggI.gather(dim=2, index=indices)
        return gI, None, ggO, None, None, None, None, None, None


class MaxPool2d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        ctx.kernel_size = _pair(kernel_size)
        ctx.stride = _pair(stride if stride is not None else kernel_size)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.ceil_mode = ceil_mode
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.SpatialDilatedMaxPooling_updateOutput(backend.library_state,
                                                      input, output, indices,
                                                      ctx.kernel_size[1], ctx.kernel_size[0],
                                                      ctx.stride[1], ctx.stride[0],
                                                      ctx.padding[1], ctx.padding[0],
                                                      ctx.dilation[1], ctx.dilation[0],
                                                      ctx.ceil_mode)
        ctx.save_for_backward(input, indices)
        ctx.mark_non_differentiable(indices)
        return output, indices

    @staticmethod
    def backward(ctx, grad_output, _indices_grad=None):
        input, indices = ctx.saved_variables
        grad_input = MaxPool2dBackward.apply(input, indices, grad_output, ctx.kernel_size, ctx.stride, ctx.padding,
                                             ctx.dilation, ctx.ceil_mode)
        return grad_input, None, None, None, None, None, None


class MaxPool2dBackward(Function):

    @staticmethod
    def forward(ctx, input, indices, grad_output, kernel_size, stride, padding, dilation,
                ceil_mode):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode

        grad_input = grad_output.new()
        ctx.save_for_backward(indices)
        backend = type2backend[type(input)]
        backend.SpatialDilatedMaxPooling_updateGradInput(backend.library_state,
                                                         input, grad_output, grad_input, indices,
                                                         ctx.kernel_size[1], ctx.kernel_size[0],
                                                         ctx.stride[1], ctx.stride[0],
                                                         ctx.padding[1], ctx.padding[0],
                                                         ctx.dilation[1], ctx.dilation[0],
                                                         ctx.ceil_mode)
        return grad_input

    @staticmethod
    def backward(ctx, ggI, _ggIndices=None):
        indices, = ctx.saved_variables

        gI = Variable(ggI.data.new(ggI.size()).zero_())
        # ggO is equivalent to the 1d case, but the indices are given wrt the last two dimensions combined
        indices_view = indices.view(indices.size()[:-2] + (-1,))
        ggO = ggI.contiguous().view(ggI.size()[:-2] + (-1,)).gather(dim=2, index=indices_view).view_as(indices)
        return gI, None, ggO, None, None, None, None, None, None


class MaxPool3d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False):
        ctx.kernel_size = _triple(kernel_size)
        ctx.stride = _triple(stride if stride is not None else kernel_size)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.ceil_mode = ceil_mode
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.VolumetricDilatedMaxPooling_updateOutput(backend.library_state,
                                                         input, output, indices,
                                                         ctx.kernel_size[0], ctx.kernel_size[2], ctx.kernel_size[1],
                                                         ctx.stride[0], ctx.stride[2], ctx.stride[1],
                                                         ctx.padding[0], ctx.padding[2], ctx.padding[1],
                                                         ctx.dilation[0], ctx.dilation[2], ctx.dilation[1],
                                                         ctx.ceil_mode)
        ctx.save_for_backward(input, indices)
        ctx.mark_non_differentiable(indices)
        return output, indices

    @staticmethod
    def backward(ctx, grad_output, _indices_grad=None):
        input, indices = ctx.saved_variables
        grad_input = MaxPool3dBackward.apply(input, indices, grad_output, ctx.kernel_size, ctx.stride,
                                             ctx.padding, ctx.dilation, ctx.ceil_mode)
        return grad_input, None, None, None, None, None, None


class MaxPool3dBackward(Function):

    @staticmethod
    def forward(ctx, input, indices, grad_output, kernel_size, stride, padding, dilation,
                ceil_mode):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.VolumetricDilatedMaxPooling_updateGradInput(backend.library_state,
                                                            input, grad_output, grad_input, indices,
                                                            ctx.kernel_size[0], ctx.kernel_size[
                                                                2], ctx.kernel_size[1],
                                                            ctx.stride[0], ctx.stride[2], ctx.stride[1],
                                                            ctx.padding[0], ctx.padding[2], ctx.padding[1],
                                                            ctx.dilation[0], ctx.dilation[2], ctx.dilation[1],
                                                            ctx.ceil_mode)
        return grad_input

    @staticmethod
    def backward(ctx, ggI, _ggIndices=None):
        raise ValueError("MaxPool3d cannot be differentiated twice")


class MaxUnpool2d(Function):

    @staticmethod
    def forward(ctx, input, indices, output_size):
        ctx.output_size = output_size
        ctx.save_for_backward(input, indices)
        ctx._backend = type2backend[type(input)]
        output = input.new()
        ctx._backend.SpatialMaxUnpooling_updateOutput(
            ctx._backend.library_state, input, output, indices,
            ctx.output_size[1], ctx.output_size[0])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, indices = ctx.saved_tensors
        grad_input = grad_output.new()
        ctx._backend.SpatialMaxUnpooling_updateGradInput(
            ctx._backend.library_state, input, grad_output, grad_input,
            indices, ctx.output_size[1], ctx.output_size[0])
        return grad_input, None, None


class MaxUnpool3d(Function):

    @staticmethod
    def forward(ctx, input, indices, output_size, stride, padding):
        ctx.output_size = output_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.save_for_backward(input, indices)
        ctx._backend = type2backend[type(input)]
        output = input.new()
        ctx._backend.VolumetricMaxUnpooling_updateOutput(
            ctx._backend.library_state, input, output, indices,
            ctx.output_size[0], ctx.output_size[2], ctx.output_size[1],
            ctx.stride[0], ctx.stride[2], ctx.stride[1],
            ctx.padding[0], ctx.padding[2], ctx.padding[1])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, indices = ctx.saved_tensors
        grad_input = grad_output.new()
        ctx._backend.VolumetricMaxUnpooling_updateGradInput(
            ctx._backend.library_state, input, grad_output, grad_input, indices,
            ctx.output_size[0], ctx.output_size[2], ctx.output_size[1],
            ctx.stride[0], ctx.stride[2], ctx.stride[1],
            ctx.padding[0], ctx.padding[2], ctx.padding[1])
        return grad_input, None, None, None, None


class FractionalMaxPool2d(Function):

    @staticmethod
    def forward(ctx, input, kh, kw, output_size=None, output_ratio=None,
                return_indices=False, _random_samples=None):
        # Pool size (how wide the pooling for each output unit is)
        ctx.kw, ctx.kh = kw, kh

        # Random samples are drawn for all
        # batch * plane * (height, width; i.e., 2) points. This determines
        # the 2d "pseudorandom" overlapping pooling regions for each
        # (batch element x input plane).
        ctx.random_samples = _random_samples

        ctx.return_indices = return_indices

        if output_size is not None:
            ctx.oh, ctx.ow = output_size
            ctx.rh, ctx.rw = None, None
        elif output_ratio is not None:
            ctx.oh, ctx.ow = None, None
            ctx.rh, ctx.rw = output_ratio
            assert 0 < ctx.rh < 1
            assert 0 < ctx.rw < 1
        else:
            assert False

        if ctx.random_samples is None:
            random_samples = input.new().resize_(input.size(0),
                                                 input.size(1), 2).uniform_()
        else:
            random_samples = ctx.random_samples
            ctx.random_samples = None

        if ctx.oh is None:
            ctx.oh = int(input.size(2) * ctx.rh)
            ctx.ow = int(input.size(3) * ctx.rw)
        assert isinstance(ctx.oh, int) and isinstance(ctx.ow, int)

        indices = input.new().long()
        output = input.new()
        ctx._backend = type2backend[type(input)]
        ctx._backend.SpatialFractionalMaxPooling_updateOutput(
            ctx._backend.library_state,
            input,
            output,
            ctx.ow, ctx.oh,
            ctx.kw, ctx.kh,
            indices,
            random_samples
        )

        ctx.random_samples = None  # Free unnecessary buffers
        if ctx.return_indices:
            ctx.save_for_backward(input, indices)
            return output, indices
        else:
            ctx.indices = indices
            ctx.save_for_backward(input)
            return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, _grad_indices=None):
        if ctx.return_indices:
            input, indices = ctx.saved_tensors
        else:
            input, = ctx.saved_tensors
            indices = ctx.indices

        grad_input = grad_output.new()
        ctx._backend.SpatialFractionalMaxPooling_updateGradInput(
            ctx._backend.library_state,
            input,
            grad_output,
            grad_input,
            ctx.ow, ctx.oh,
            ctx.kw, ctx.kh,
            indices)

        return grad_input, None, None, None, None, None, None


class AvgPool2d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0,
                ceil_mode=False, count_include_pad=True):
        ctx.kernel_size = _pair(kernel_size)
        ctx.stride = _pair(stride if stride is not None else kernel_size)
        ctx.padding = _pair(padding)
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad
        backend = type2backend[type(input)]
        output = input.new()
        # can avoid this with cudnn
        ctx.save_for_backward(input)
        backend.SpatialAveragePooling_updateOutput(
            backend.library_state,
            input, output,
            ctx.kernel_size[1], ctx.kernel_size[0],
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.ceil_mode, ctx.count_include_pad)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = AvgPool2dBackward.apply(input, grad_output, ctx.kernel_size, ctx.stride,
                                             ctx.padding, ctx.ceil_mode, ctx.count_include_pad)
        return grad_input, None, None, None, None, None


class AvgPool2dBackward(Function):

    @staticmethod
    def forward(ctx, input, grad_output, kernel_size, stride, padding, ceil_mode, count_include_pad):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad
        backend = type2backend[type(grad_output)]
        grad_input = grad_output.new()
        ctx.save_for_backward(input)
        backend.SpatialAveragePooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input,
            ctx.kernel_size[1], ctx.kernel_size[0],
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.ceil_mode, ctx.count_include_pad)
        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        input, = ctx.saved_variables
        gI = Variable(ggI.data.new(ggI.size()).zero_())
        ggO = AvgPool2d.apply(ggI, ctx.kernel_size, ctx.stride, ctx.padding, ctx.ceil_mode, ctx.count_include_pad)
        return gI, ggO, None, None, None, None, None


class AvgPool3d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride=None):
        ctx.kernel_size = _triple(kernel_size)
        ctx.stride = _triple(stride if stride is not None else kernel_size)
        backend = type2backend[type(input)]
        output = input.new()
        # can avoid this with cudnn
        ctx.save_for_backward(input)
        backend.VolumetricAveragePooling_updateOutput(backend.library_state,
                                                      input, output,
                                                      ctx.kernel_size[0], ctx.kernel_size[2], ctx.kernel_size[1],
                                                      ctx.stride[0], ctx.stride[2], ctx.stride[1])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = AvgPool3dBackward.apply(input, grad_output, ctx.kernel_size, ctx.stride)
        return grad_input, None, None


class AvgPool3dBackward(Function):

    @staticmethod
    def forward(ctx, input, grad_output, kernel_size, stride):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        backend = type2backend[type(grad_output)]
        grad_input = grad_output.new()
        ctx.save_for_backward(input)
        backend.VolumetricAveragePooling_updateGradInput(backend.library_state,
                                                         input, grad_output, grad_input,
                                                         ctx.kernel_size[0], ctx.kernel_size[2], ctx.kernel_size[1],
                                                         ctx.stride[0], ctx.stride[2], ctx.stride[1])
        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        input, = ctx.saved_variables
        gI = Variable(ggI.data.new(ggI.size()).zero_())
        ggO = AvgPool3d.apply(ggI, ctx.kernel_size, ctx.stride)
        return gI, ggO, None, None


class AdaptiveMaxPool1d(Function):

    @staticmethod
    def forward(ctx, input, output_size, return_indices=False):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

        ctx.output_size = _single(output_size)
        ctx.return_indices = return_indices
        input2d = input.unsqueeze(2)    # size = N*C*1*L
        backend = type2backend[type(input)]
        indices, output = input2d.new().long(), input2d.new()
        backend.SpatialAdaptiveMaxPooling_updateOutput(backend.library_state,
                                                       input2d, output, indices,
                                                       ctx.output_size[0], 1)
        indices = indices.squeeze(2)
        output = output.squeeze(2)
        if ctx.return_indices:
            ctx.save_for_backward(input, indices)
            ctx.mark_non_differentiable(indices)
            return output, indices
        else:
            ctx.save_for_backward(input)
            ctx.indices = indices
            return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, _indices_grad=None):
        if ctx.return_indices:
            input, indices = ctx.saved_tensors
        else:
            input, = ctx.saved_tensors
            indices = ctx.indices

        input2d = input.unsqueeze(2)
        indices2d = indices.unsqueeze(2)
        grad_output2d = grad_output.unsqueeze(2)
        grad_input = grad_output2d.new()
        backend = type2backend[type(input)]
        backend.SpatialAdaptiveMaxPooling_updateGradInput(backend.library_state,
                                                          input2d, grad_output2d, grad_input, indices2d)
        grad_input = grad_input.squeeze(2)
        return grad_input, None, None


class AdaptiveMaxPool2d(Function):

    @staticmethod
    def forward(ctx, input, output_size, return_indices=False):
        ctx.output_size = _pair(output_size)
        ctx.return_indices = return_indices
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.SpatialAdaptiveMaxPooling_updateOutput(backend.library_state,
                                                       input, output, indices,
                                                       ctx.output_size[1], ctx.output_size[0])
        if ctx.return_indices:
            ctx.save_for_backward(input, indices)
            ctx.mark_non_differentiable(indices)
            return output, indices
        else:
            ctx.save_for_backward(input)
            ctx.indices = indices
            return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, _indices_grad=None):
        if ctx.return_indices:
            input, indices = ctx.saved_tensors
        else:
            input, = ctx.saved_tensors
            indices = ctx.indices
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialAdaptiveMaxPooling_updateGradInput(backend.library_state,
                                                          input, grad_output, grad_input, indices)
        return grad_input, None, None


class AdaptiveAvgPool1d(Function):

    @staticmethod
    def forward(ctx, input, output_size):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

        ctx.output_size = _single(output_size)
        input2d = input.unsqueeze(2)    # size = N*C*1*L
        backend = type2backend[type(input)]
        output = input2d.new()
        ctx.save_for_backward(input)
        backend.SpatialAdaptiveAveragePooling_updateOutput(
            backend.library_state,
            input2d, output,
            ctx.output_size[0], 1)
        output = output.squeeze(2)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        backend = type2backend[type(grad_output)]
        input, = ctx.saved_tensors
        input2d = input.unsqueeze(2)
        grad_output2d = grad_output.unsqueeze(2)
        grad_input = grad_output2d.new()
        backend.SpatialAdaptiveAveragePooling_updateGradInput(
            backend.library_state,
            input2d, grad_output2d, grad_input)
        grad_input = grad_input.squeeze(2)
        return grad_input, None


class AdaptiveAvgPool2d(Function):

    @staticmethod
    def forward(ctx, input, output_size):
        ctx.output_size = _pair(output_size)
        backend = type2backend[type(input)]
        output = input.new()
        ctx.save_for_backward(input)
        backend.SpatialAdaptiveAveragePooling_updateOutput(
            backend.library_state,
            input, output,
            ctx.output_size[1], ctx.output_size[0])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        backend = type2backend[type(grad_output)]
        input, = ctx.saved_tensors
        grad_input = grad_output.new()
        backend.SpatialAdaptiveAveragePooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input)
        return grad_input, None


class Mac1d(Function):

    def forward(self, input):
        input_len = input.size(2)
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        self.save_for_backward(input)
        backend.SpatialDilatedMaxPooling_updateOutput(
            backend.library_state,
            input, output, indices,
            input_len, 1,  # kernel size
            input_len, 1,  # stride
            0, 0,  # padding
            1, 1,  # dilation
            False)
        self.save_for_backward(input)
        self.indices = indices
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        indices = self.indices
        input_len = input.size(2)
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialDilatedMaxPooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input, indices,
            input_len, 1,  # kernel size
            input_len, 1,  # stride
            0, 0,  # padding
            1, 1,  # dilation
            False)
        return grad_input


class Mac2d(Function):

    def forward(self, input):
        input_height, input_width = input.size(2), input.size(3)
        pool_size = min(input_height, input_width)
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        self.save_for_backward(input)
        backend.SpatialDilatedMaxPooling_updateOutput(
            backend.library_state,
            input, output, indices,
            pool_size, pool_size,  # kernel size
            pool_size, pool_size,  # stride
            0, 0,  # padding
            1, 1,  # dilation
            False)
        self.save_for_backward(input)
        self.indices = indices
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        indices = self.indices
        input_height, input_width = input.size(2), input.size(3)
        pool_size = min(input_height, input_width)
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialDilatedMaxPooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input, indices,
            pool_size, pool_size,  # kernel size
            pool_size, pool_size,  # stride
            0, 0,  # padding
            1, 1,  # dilation
            False)
        return grad_input


class Rmac1d(Function):

    def __init__(self, levels=3, overlap=0.4, eps=1e-4):
        self.levels = levels
        self.overlap = overlap
        self.eps = eps

    def forward(self, input):
        batch_size, num_features, input_len = input.size()
        eps = self.eps
        levels = self.levels

        backend = type2backend[type(input)]
        all_indices = []
        all_output = []
        self.save_for_backward(input)

        for level in range(levels):
            output = input.new()
            indices = input.new().long()
            pool_size = 2 * input_len // (level + 2)
            stride = (input_len - pool_size) // (level or inf)
            stride = stride or pool_size
            backend.SpatialDilatedMaxPooling_updateOutput(
                backend.library_state,
                input, output, indices,
                pool_size, 1,  # kernel size
                stride, 1,  # stride
                0, 0,  # padding
                1, 1,  # dilation
                False)
            # output = output.view(batch_size, num_features, -1)

            all_indices.append(indices)
            all_output.append(output)

        all_output = cat(all_output, 2)
        region_norms = all_output.norm(2,1).expand_as(all_output) + eps
        all_output = all_output.div(region_norms).sum(2).squeeze()
        self.all_indices = all_indices

        # Necessary as the output becomes 1 everywhere when dividing by
        # torch.norm(output, 2, 0) if batchsize is one
        if batch_size == 1:
            batch_norm = all_output.norm(2) + eps
        else:
            batch_norm = all_output.norm(2, 0).expand_as(all_output) + eps
        return all_output / batch_norm

    def backward(self, all_grad_output):
        input, = self.saved_tensors
        input_len = input.size(2)
        levels = self.levels

        all_grad_output = all_grad_output.unsqueeze(2)
        backend = type2backend[type(input)]
        all_indices = self.all_indices
        all_grad_input = all_grad_output.new()

        for level in range(levels):
            grad_input = all_grad_output.new()
            indices = all_indices[level]
            grad_output = all_grad_output.expand_as(indices.squeeze(0))
            pool_size = 2 * input_len // (level + 2)
            stride = (input_len - pool_size) // (level or inf)
            stride = stride or pool_size
            backend.SpatialDilatedMaxPooling_updateGradInput(
                backend.library_state,
                input, grad_output, grad_input, indices,
                pool_size, 1,  # kernel size
                stride, 1,  # stride
                0, 0,  # padding
                1, 1,  # dilation
                False)
            if all_grad_input.dim() == 0:
                all_grad_input = grad_input
            else:
                all_grad_input += grad_input
        return all_grad_input


class Rmac2d(Function):

    def __init__(self, levels=3, overlap=0.4, eps=1e-4):
        self.levels = levels
        self.overlap = overlap
        self.eps = eps
        if cuda.is_available():
            self.steps = arange(0, 20).cuda()
        else:
            self.steps = arange(0, 20)

    def _ratio2regions(self, input_width, input_height): # needs rename (no idea)
        if input_width != input_height:
            small_edge = min(input_width, input_height)
            large_edge = max(input_height, input_width)
            max_steps = large_edge // small_edge
            overlap = self.overlap
            eps = self.eps
            steps = self.steps.narrow(0, 0, max_steps)
            b = (large_edge - small_edge) / (steps + 1)
            val = ((small_edge**2 - small_edge * b) / (small_edge**2) - overlap).abs()
            idx = int(steps.dot(val.eq(val.min()).float()))
            if input_height < input_width:
                w_steps, h_steps = idx, 0
            elif input_height > input_width:
                w_steps, h_steps = 0, idx
        else:
            w_steps, h_steps = 0, 0
        return w_steps, h_steps

    def forward(self, input):
        batch_size, num_features, input_height, input_width = input.size()
        small_edge = min(input_height, input_width)
        w_steps, h_steps = self._ratio2regions(input_height, input_width)
        eps = self.eps
        levels = self.levels

        backend = type2backend[type(input)]
        all_indices = []
        all_output = []
        self.save_for_backward(input)

        for level in range(levels):
            output = input.new()
            indices = input.new().long()
            pool_size = 2 * small_edge // (level + 2)
            w_stride = (input_width - pool_size) // (level + w_steps or inf)
            w_stride = w_stride or pool_size
            h_stride = (input_height - pool_size) // (level + h_steps or inf)
            h_stride = h_stride or pool_size
            backend.SpatialDilatedMaxPooling_updateOutput(
                backend.library_state,
                input, output, indices,
                pool_size, pool_size,  # kernel size
                w_stride, h_stride,  # stride
                0, 0,  # padding
                1, 1,  # dilation
                False)

            all_indices.append(indices)
            output = output.view(batch_size, num_features, -1)
            all_output.append(output)

        all_output = cat(all_output, 2)
        region_norms = all_output.norm(2,1).expand_as(all_output) + eps
        all_output = all_output.div(region_norms).sum(2).squeeze()
        self.all_indices = all_indices

        # Necessary as the output becomes 1 everywhere when dividing by
        # torch.norm(output, 2, 0) if batchsize is one
        if batch_size == 1:
            batch_norm = all_output.norm(2) + eps
        else:
            batch_norm = all_output.norm(2, 0).expand_as(all_output) + eps
        return all_output / batch_norm

    def backward(self, all_grad_output):
        input, = self.saved_tensors
        input_height, input_width = input.size(2), input.size(3)
        small_edge = min(input_height, input_width)
        w_steps, h_steps = self._ratio2regions(input_height, input_width)
        levels = self.levels

        all_grad_output = all_grad_output.unsqueeze(2).unsqueeze(2)
        backend = type2backend[type(input)]
        all_indices = self.all_indices
        all_grad_input = all_grad_output.new()

        for level in range(levels):
            grad_input = all_grad_output.new()
            indices = all_indices[level]
            grad_output = all_grad_output.expand_as(indices)
            pool_size = 2 * small_edge // (level + 2)
            w_stride = (input_width - pool_size) // (level + w_steps or inf)
            w_stride = w_stride or pool_size
            h_stride = (input_height - pool_size) // (level + h_steps or inf)
            h_stride = h_stride or pool_size
            backend.SpatialDilatedMaxPooling_updateGradInput(
                backend.library_state,
                input, grad_output, grad_input, indices,
                pool_size, pool_size,  # kernel size
                w_stride, h_stride,  # stride
                0, 0,  # padding
                1, 1,  # dilation
                False)
            if all_grad_input.dim() == 0:
                all_grad_input = grad_input
            else:
                all_grad_input += grad_input
        return all_grad_input


class Aac1d(Function):

    def forward(self, input):
        input_len = input.size(2)
        backend = type2backend[type(input)]
        output = input.new()
        self.save_for_backward(input)
        backend.SpatialAveragePooling_updateOutput(
            backend.library_state,
            input, output,
            input_len, 1,  # kernel size
            input_len, 1,  # stride
            0, 0,  # padding
            False, False)
        self.save_for_backward(input)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        input_len = input.size(2)
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialAveragePooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input,
            input_len, 1,  # kernel size
            input_len, 1,  # stride
            0, 0,  # padding
            False, False)
        return grad_input


class Aac2d(Function):

    def forward(self, input):
        input_height, input_width = input.size(2), input.size(3)
        pool_size = min(input_height, input_width)
        backend = type2backend[type(input)]
        output = input.new()
        self.save_for_backward(input)
        backend.SpatialAveragePooling_updateOutput(
            backend.library_state,
            input, output,
            pool_size, pool_size,  # kernel size
            pool_size, pool_size,  # stride
            0, 0,  # padding
            False, False)
        self.save_for_backward(input)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        input_height, input_width = input.size(2), input.size(3)
        pool_size = min(input_height, input_width)
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialAveragePooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input,
            pool_size, pool_size,  # kernel size
            pool_size, pool_size,  # stride
            0, 0,  # padding
            False, False)
        return grad_input


class Raac1d(Function):

    def __init__(self, levels=3, overlap=0.4, eps=1e-4):
        self.levels = levels
        self.overlap = overlap
        self.eps = eps

    def forward(self, input):
        batch_size, num_features, input_len = input.size()
        eps = self.eps
        levels = self.levels

        backend = type2backend[type(input)]
        all_sizes = [] # needed in replacement of all_indices hook
        all_output = []
        self.save_for_backward(input)

        for level in range(levels):
            output = input.new()
            pool_size = 2 * input_len // (level + 2)
            stride = (input_len - pool_size) // (level or inf)
            stride = stride or pool_size
            backend.SpatialAveragePooling_updateOutput(
                backend.library_state,
                input, output,
                pool_size, 1,  # kernel size
                stride, 1,  # stride
                0, 0,  # padding
                False, False)
            # output = output.view(batch_size, num_features, -1)

            all_sizes.append(output.size())
            all_output.append(output)

        all_output = cat(all_output, 2)
        region_norms = all_output.norm(2,1).expand_as(all_output) + eps
        all_output = all_output.div(region_norms).sum(2).squeeze()
        self.all_sizes = all_sizes

        # Necessary as the output becomes 1 everywhere when dividing by
        # torch.norm(output, 2, 0) if batchsize is one
        if batch_size == 1:
            batch_norm = all_output.norm(2) + eps
        else:
            batch_norm = all_output.norm(2, 0).expand_as(all_output) + eps
        return all_output / batch_norm

    def backward(self, all_grad_output):
        input, = self.saved_tensors
        input_len = input.size(2)
        levels = self.levels

        all_grad_output = all_grad_output.unsqueeze(2)
        backend = type2backend[type(input)]
        all_sizes = self.all_sizes
        all_grad_input = all_grad_output.new()

        for level in range(levels):
            grad_input = all_grad_output.new()
            current_size = all_sizes[level]
            print(all_grad_output.size(), current_size)
            grad_output = all_grad_output.expand(current_size)
            pool_size = 2 * input_len // (level + 2)
            stride = (input_len - pool_size) // (level or inf)
            stride = stride or pool_size
            backend.SpatialAveragePooling_updateGradInput(
                backend.library_state,
                input, grad_output, grad_input,
                pool_size, 1,  # kernel size
                stride, 1,  # stride
                0, 0,  # padding
                False, False)
            if all_grad_input.dim() == 0:
                all_grad_input = grad_input
            else:
                all_grad_input += grad_input
        return all_grad_input


class Raac2d(Function):

    def __init__(self, levels=3, overlap=0.4, eps=1e-4):
        self.levels = levels
        self.overlap = overlap
        self.eps = eps
        if cuda.is_available():
            self.steps = arange(0, 20).cuda()
        else:
            self.steps = arange(0, 20)

    def _ratio2regions(self, input_width, input_height): # needs rename (no idea)
        if input_width != input_height:
            small_edge = min(input_width, input_height)
            large_edge = max(input_height, input_width)
            max_steps = large_edge // small_edge
            overlap = self.overlap
            eps = self.eps
            steps = self.steps.narrow(0, 0, max_steps)
            b = (large_edge - small_edge) / (steps + 1)
            val = ((small_edge**2 - small_edge * b) / (small_edge**2) - overlap).abs()
            idx = int(steps.dot(val.eq(val.min()).float()))
            if input_height < input_width:
                w_steps, h_steps = idx, 0
            elif input_height > input_width:
                w_steps, h_steps = 0, idx
        else:
            w_steps, h_steps = 0, 0
        return w_steps, h_steps

    def forward(self, input):
        batch_size, num_features, input_height, input_width = input.size()
        small_edge = min(input_height, input_width)
        w_steps, h_steps = self._ratio2regions(input_height, input_width)
        eps = self.eps
        levels = self.levels

        backend = type2backend[type(input)]
        all_sizes = []
        all_output = []
        self.save_for_backward(input)

        for level in range(levels):
            output = input.new()
            pool_size = 2 * small_edge // (level + 2)
            w_stride = (input_width - pool_size) // (level + w_steps or inf)
            w_stride = w_stride or pool_size
            h_stride = (input_height - pool_size) // (level + h_steps or inf)
            h_stride = h_stride or pool_size
            backend.SpatialAveragePooling_updateOutput(
                backend.library_state,
                input, output,
                pool_size, pool_size,  # kernel size
                w_stride, h_stride,  # stride
                0, 0,  # padding
                False, False)

            all_sizes.append(output.size())
            output = output.view(batch_size, num_features, -1)
            all_output.append(output)

        all_output = cat(all_output, 2)
        region_norms = all_output.norm(2,1).expand_as(all_output) + eps
        all_output = all_output.div(region_norms).sum(2).squeeze(2)
        self.all_sizes = all_sizes

        # Necessary as the output becomes 1 everywhere when dividing by
        # torch.norm(output, 2, 0) if batchsize is one
        if batch_size == 1:
            batch_norm = all_output.norm(2) + eps
        else:
            batch_norm = all_output.norm(2, 0).expand_as(all_output) + eps
        return all_output / batch_norm

    def backward(self, all_grad_output):
        input, = self.saved_tensors
        input_height, input_width = input.size(2), input.size(3)
        small_edge = min(input_height, input_width)
        w_steps, h_steps = self._ratio2regions(input_height, input_width)
        levels = self.levels

        all_grad_output = all_grad_output.unsqueeze(2).unsqueeze(2)
        backend = type2backend[type(input)]
        all_sizes = self.all_sizes
        all_grad_input = all_grad_output.new()

        for level in range(levels):
            grad_input = all_grad_output.new()
            current_size = all_sizes[level]
            grad_output = all_grad_output.expand(current_size)
            pool_size = 2 * small_edge // (level + 2)
            w_stride = (input_width - pool_size) // (level + w_steps or inf)
            w_stride = w_stride or pool_size
            h_stride = (input_height - pool_size) // (level + h_steps or inf)
            h_stride = h_stride or pool_size
            backend.SpatialAveragePooling_updateGradInput(
                backend.library_state,
                input, grad_output, grad_input,
                pool_size, pool_size,  # kernel size
                w_stride, h_stride,  # stride
                0, 0,  # padding
                False, False)
            if all_grad_input.dim() == 0:
                all_grad_input = grad_input
            else:
                all_grad_input += grad_input
        return all_grad_input


_all_functions.append(AvgPool2d)
_all_functions.append(AvgPool2dBackward)
_all_functions.append(AvgPool3d)
_all_functions.append(AvgPool3dBackward)
_all_functions.append(MaxPool1d)
_all_functions.append(MaxPool1dBackward)
_all_functions.append(MaxPool2d)
_all_functions.append(MaxPool2dBackward)
_all_functions.append(MaxPool3d)
_all_functions.append(MaxPool3dBackward)
_all_functions.append(MaxUnpool2d)
_all_functions.append(MaxUnpool3d)
_all_functions.append(FractionalMaxPool2d)
_all_functions.append(AdaptiveMaxPool1d)
_all_functions.append(AdaptiveMaxPool2d)
_all_functions.append(AdaptiveAvgPool1d)
_all_functions.append(AdaptiveAvgPool2d)
_all_functions.append(Mac1d)
_all_functions.append(Mac2d)
_all_functions.append(Rmac1d)
_all_functions.append(Rmac2d)
_all_functions.append(Aac1d)
_all_functions.append(Aac2d)
_all_functions.append(Raac1d)
_all_functions.append(Raac2d)
