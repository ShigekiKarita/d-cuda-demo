import std.stdio : writeln, writefln;
import std.format : format;
import std.conv : to;
import std.file : readText;
import std.string : toStringz, fromStringz;

import derelict.cuda;


void checkCudaErrors(CUresult err) {
    const(char)* name, content;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &content);

    assert(err == CUDA_SUCCESS, name.fromStringz ~ ": " ~ content.fromStringz);
}


void main()
{
    DerelictCUDADriver.load();
    CUdevice device;
    CUcontext context;

    // Initialize the driver API
    cuInit(0);
    // Get a handle to the first compute device
    cuDeviceGet(&device, 0);
    // Create a compute device context
    cuCtxCreate(&context, 0, device);

    CUmodule cuModule;
    CUfunction cuFunction;
    auto ptxstr = readText("kernel/saxpy.ptx");
    writeln(ptxstr);

    // JIT compile a null-terminated PTX string
    checkCudaErrors(cuModuleLoadData(&cuModule, cast(void*) ptxstr.toStringz));

    // Get a handle to the "myfunction" kernel function
    checkCudaErrors(cuModuleGetFunction(&cuFunction, cuModule, "saxpy"));


    // Device data
    CUdeviceptr devBufferA;
    CUdeviceptr devBufferB;
    CUdeviceptr devBufferC;
    size_t n = 16;

    checkCudaErrors(cuMemAlloc(&devBufferA, float.sizeof * n));
    checkCudaErrors(cuMemAlloc(&devBufferB, float.sizeof * n));
    checkCudaErrors(cuMemAlloc(&devBufferC, float.sizeof * n));

    auto hostA = new float[n];
    auto hostB = new float[n];
    auto hostC = new float[n];

    // Populate input
    for (size_t i = 0; i != n; ++i) {
        hostA[i] = cast(float)i;
        hostB[i] = cast(float)(2*i);
        hostC[i] = 0.0f;
    }

    checkCudaErrors(cuMemcpyHtoD(devBufferA, &hostA[0], float.sizeof * n));
    checkCudaErrors(cuMemcpyHtoD(devBufferB, &hostB[0], float.sizeof * n));

    // Kernel parameters
    void*[5] params = null;
    params[0] = &devBufferC;
    params[1] = &devBufferA;
    params[2] = &devBufferB;
    params[3] = &n;
    // auto KernelParams = cast(void*[]) [ &devBufferC, 1.0, &devBufferA, &devBufferB, 16 ];
    // Kernel launch
    checkCudaErrors(cuLaunchKernel(cuFunction,
                                   1U, 1U, 1U, // grids
                                   16U, 1U, 1U, // blocks
                                   0, null, params.ptr, null));


    // Retrieve device data
    checkCudaErrors(cuMemcpyDtoH(&hostC[0], devBufferC, float.sizeof * n));

    foreach (i; 0 .. n) {
        writefln!"%f + %f = %f"(hostA[i], hostB[i], hostC[i]);
        assert(hostA[i] + hostB[i] == hostC[i]);
    }

    // Clean-up
    checkCudaErrors(cuMemFree(devBufferA));
    checkCudaErrors(cuMemFree(devBufferB));
    checkCudaErrors(cuMemFree(devBufferC));
    checkCudaErrors(cuModuleUnload(cuModule));
    checkCudaErrors(cuCtxDestroy(context));
}
