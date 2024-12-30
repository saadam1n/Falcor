import torch
import slangtorch

module = slangtorch.loadModule("./prototyping/bilateral_filter.slang", verbose=False)

class BilateralFilter(torch.autograd.Function):
    @staticmethod # We can accept as many parameters here as we wish
    def forward(ctx, input: torch.Tensor, params: torch.Tensor, kernel_size: int, dialation: int):
        if(input.dim() < 4):
            input.unsqueeze(0)

        if(input.size(1) != params.size(0) - 2):
            raise RuntimeError(f"Size mismatch for channels. Expected {params.size(0) - 2} but input shape was {input.shape}")

        output = torch.zeros_like(input)

        numPixels = input.size(0) * input.size(2) * input.size(3)

        blockSize = 768
        numBlocks = numPixels // blockSize
        module.exec_bilateral_filter_wrapper(
            input = input,
            params = params,
            output = output,
            kernel_boundary = kernel_size // 2,
            dialation = dialation
        ).launchRaw(
            blockSize=(blockSize, 1, 1),
            gridSize=(int(numBlocks), 1, 1)
        )

        ctx.save_for_backward(input, params)
        ctx.kernel_size = kernel_size
        ctx.dialation = dialation

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input, params) = ctx.saved_tensors

        kernel_size = ctx.kernel_size
        dialation = ctx.dialation

        input_grad = torch.zeros_like(input)
        params_grad = torch.zeros_like(params)
        output = torch.zeros_like(input)


        numPixels = input.size(0) * input.size(2) * input.size(3)
        blockSize = 768
        numBlocks = numPixels // blockSize
        module.bwd_bilateral_filter_wrapper(
            input = input,
            input_grad = input_grad,
            params = params,
            params_grad = params_grad,
            output = output,
            output_grad = grad_output,
            kernel_boundary = kernel_size // 2,
            dialation = dialation
        ).launchRaw(
            blockSize=(blockSize, 1, 1),
            gridSize=(numBlocks, 1, 1)
        )

        params_grad.nan_to_num_(nan=0.0, posinf=100.0, neginf=100.0)

        return input_grad, params_grad, None, None

    @staticmethod
    def symbolic(g, input, params, kernel_size, dialation):
        return g.op(
            "BilateralFilter",
            input,
            params,
            kernel_size,
            dialation
        )

    @staticmethod
    def _(input, params, kernel_size, dilation):
        torch._check(input.size(1) + 2 == params.size(0))
        torch._check(input.dtype == torch.float)
        torch._check(params.dtype == torch.float)
        return torch.empty_like(input)



class KernelBilateralFilter(torch.autograd.Function):
    @staticmethod # We can accept as many parameters here as we wish
    def forward(ctx, input: torch.Tensor, params: torch.Tensor, kernel: torch.Tensor, dialation: int):
        if(input.dim() < 4):
            input.unsqueeze(0)

        if(input.size(1) != params.size(0) - 2):
            raise RuntimeError(f"Size mismatch for channels. Expected {params.size(0) - 2} but input shape was {input.shape}")

        output = torch.zeros_like(input)

        numPixels = input.size(0) * input.size(2) * input.size(3)

        kernel_size = kernel.size(0)

        blockSize = 768
        numBlocks = numPixels // blockSize
        module.exec_kernel_bilateral_filter_wrapper(
            input = input,
            params = params,
            kernel = kernel,
            output = output,
            kernel_boundary = kernel_size // 2,
            dialation = dialation
        ).launchRaw(
            blockSize=(blockSize, 1, 1),
            gridSize=(int(numBlocks), 1, 1)
        )

        ctx.save_for_backward(input, params, kernel)
        ctx.kernel_size = kernel_size
        ctx.dialation = dialation

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input, params, kernel) = ctx.saved_tensors

        kernel_size = ctx.kernel_size
        dialation = ctx.dialation

        input_grad = torch.zeros_like(input)
        params_grad = torch.zeros_like(params)
        kernel_grad = torch.zeros_like(kernel)
        output = torch.zeros_like(input)


        numPixels = input.size(0) * input.size(2) * input.size(3)
        blockSize = 768
        numBlocks = numPixels // blockSize
        module.bwd_kernel_bilateral_filter_wrapper(
            input = input,
            input_grad = input_grad,
            params = params,
            params_grad = params_grad,
            kernel = kernel,
            kernel_grad = kernel_grad,
            output = output,
            output_grad = grad_output,
            kernel_boundary = kernel_size // 2,
            dialation = dialation
        ).launchRaw(
            blockSize=(blockSize, 1, 1),
            gridSize=(numBlocks, 1, 1)
        )

        params_grad.nan_to_num_(nan=0.0, posinf=100.0, neginf=100.0)

        return input_grad, params_grad, None, None

    @staticmethod
    def symbolic(g, input, params, kernel_size, dialation):
        return g.op(
            "KernelBilateralFilter",
            input,
            params,
            kernel_size,
            dialation
        )

    @staticmethod
    def _(input, params, kernel_size, dilation):
        torch._check(input.size(1) + 2 == params.size(0))
        torch._check(input.dtype == torch.float)
        torch._check(params.dtype == torch.float)
        return torch.empty_like(input)

class PixelBilateralFilter(torch.autograd.Function):
    @staticmethod # We can accept as many parameters here as we wish
    def forward(ctx, input: torch.Tensor, params: torch.Tensor, kernel_size: int, dialation: int):
        if(input.dim() < 4):
            input.unsqueeze(0)

        if(input.size(1) != params.size(1) - 2):
            raise RuntimeError(f"Size mismatch for channels. Expected {params.size(1) - 2} but input shape was {input.shape}")

        if(input.size(0) != params.size(0) or input.size(2) != params.size(2) or input.size(3) != params.size(3)):
            raise RuntimeError(f"Dimensions of input and parameters do not match. Input shape was {input.shape} while parameter shape was {params.shape}")


        output = torch.zeros_like(input)

        numPixels = input.size(0) * input.size(2) * input.size(3)

        blockSize = 768
        numBlocks = numPixels // blockSize
        module.exec_pixel_bilateral_filter_wrapper(
            input = input,
            params = params,
            output = output,
            kernel_boundary = kernel_size // 2,
            dialation = dialation
        ).launchRaw(
            blockSize=(blockSize, 1, 1),
            gridSize=(int(numBlocks), 1, 1)
        )

        ctx.save_for_backward(input, params)
        ctx.kernel_size = kernel_size
        ctx.dialation = dialation

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input, params) = ctx.saved_tensors

        kernel_size = ctx.kernel_size
        dialation = ctx.dialation

        input_grad = torch.zeros_like(input)
        params_grad = torch.zeros_like(params)
        output = torch.zeros_like(input)


        numPixels = input.size(0) * input.size(2) * input.size(3)
        blockSize = 768
        numBlocks = numPixels // blockSize
        module.bwd_pixel_bilateral_filter_wrapper(
            input = input,
            input_grad = input_grad,
            params = params,
            params_grad = params_grad,
            output = output,
            output_grad = grad_output,
            kernel_boundary = kernel_size // 2,
            dialation = dialation
        ).launchRaw(
            blockSize=(blockSize, 1, 1),
            gridSize=(numBlocks, 1, 1)
        )

        params_grad.nan_to_num_(nan=0.0, posinf=100.0, neginf=100.0)

        return input_grad, params_grad, None, None

    @staticmethod
    def symbolic(g, input, params, kernel_size, dialation):
        return g.op(
            "BilateralFilter",
            input,
            params,
            kernel_size,
            dialation
        )

    @staticmethod
    def _(input, params, kernel_size, dilation):
        torch._check(input.size(1) + 2 == params.size(1))
        torch._check(input.dtype == torch.float)
        torch._check(params.dtype == torch.float)
        return torch.empty_like(input)
