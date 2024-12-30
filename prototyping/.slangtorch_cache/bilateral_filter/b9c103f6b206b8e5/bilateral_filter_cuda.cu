#define SLANG_PRELUDE_EXPORT

#ifdef __CUDACC_RTC__
#define SLANG_CUDA_RTC 1
#else
#define SLANG_CUDA_RTC 0
#endif

#if SLANG_CUDA_RTC

#else

#include <cstdint>
#include <stdio.h>

#endif

// Define SLANG_CUDA_ENABLE_HALF to use the cuda_fp16 include to add half support.
// For this to work NVRTC needs to have the path to the CUDA SDK.
//
// As it stands the includes paths defined for Slang are passed down to NVRTC. Similarly defines
// defined for the Slang compile are passed down.

#ifdef SLANG_CUDA_ENABLE_HALF
// We don't want half2 operators, because it will implement comparison operators that return a
// bool(!). We want to generate those functions. Doing so means that we will have to define all
// the other half2 operators.
#define __CUDA_NO_HALF2_OPERATORS__
#include <cuda_fp16.h>
#endif

#ifdef SLANG_CUDA_ENABLE_OPTIX
#include <optix.h>
#endif

// Define slang offsetof implementation
#ifndef SLANG_OFFSET_OF
#define SLANG_OFFSET_OF(type, member) (size_t)((char*)&(((type*)0)->member) - (char*)0)
#endif

#ifndef SLANG_ALIGN_OF
#define SLANG_ALIGN_OF(type) __alignof__(type)
#endif

// Must be large enough to cause overflow and therefore infinity
#ifndef SLANG_INFINITY
#define SLANG_INFINITY ((float)(1e+300 * 1e+300))
#endif

// For now we'll disable any asserts in this prelude
#define SLANG_PRELUDE_ASSERT(x)

#ifndef SLANG_CUDA_WARP_SIZE
#define SLANG_CUDA_WARP_SIZE 32
#endif

#define SLANG_CUDA_WARP_MASK \
    (SLANG_CUDA_WARP_SIZE - 1) // Used for masking threadIdx.x to the warp lane index
#define SLANG_CUDA_WARP_BITMASK (~int(0))

//
#define SLANG_FORCE_INLINE inline

#define SLANG_CUDA_CALL __device__

#define SLANG_FORCE_INLINE inline
#define SLANG_INLINE inline


// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ZERO_INDEX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

// This macro handles how out-of-range surface coordinates are handled;
// I can equal
// cudaBoundaryModeClamp, in which case out-of-range coordinates are clamped to the valid range
// cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are
// ignored cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to
// fail.

#ifndef SLANG_CUDA_BOUNDARY_MODE
#define SLANG_CUDA_BOUNDARY_MODE cudaBoundaryModeZero

// Can be one of SLANG_CUDA_PTX_BOUNDARY_MODE. Only applies *PTX* emitted CUDA operations
// which currently is just RWTextureRW format writes
//
// .trap         causes an execution trap on out-of-bounds addresses
// .clamp        stores data at the nearest surface location (sized appropriately)
// .zero         drops stores to out-of-bounds addresses

#define SLANG_PTX_BOUNDARY_MODE "zero"
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

// Typically defined in cuda.h, but we can't ship/rely on that, so just define here
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;

// On CUDA sampler state is actually bound up with the texture object. We have a SamplerState type,
// backed as a pointer, to simplify code generation, with the downside that such a binding will take
// up uniform space, even though it will have no effect.
// TODO(JS): Consider ways to strip use of variables of this type so have no binding,
struct SamplerStateUnused;
typedef SamplerStateUnused* SamplerState;


// TODO(JS): Not clear yet if this can be handled on CUDA, by just ignoring.
// For now, just map to the index type.
typedef size_t NonUniformResourceIndex;

// Code generator will generate the specific type
template<typename T, int ROWS, int COLS>
struct Matrix;

typedef int1 bool1;
typedef int2 bool2;
typedef int3 bool3;
typedef int4 bool4;

#if SLANG_CUDA_RTC

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#endif

typedef long long longlong;
typedef unsigned long long ulonglong;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

union Union32
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float make_float(T val)
{
    return (float)val;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float _slang_fmod(float x, float y)
{
    return ::fmodf(x, y);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double _slang_fmod(double x, double y)
{
    return ::fmod(x, y);
}

#if SLANG_CUDA_ENABLE_HALF

// Add the other vector half types
struct __half1
{
    __half x;
};
struct __align__(4) __half3
{
    __half x, y, z;
};
struct __align__(4) __half4
{
    __half x, y, z, w;
};
#endif

#define SLANG_VECTOR_GET_ELEMENT(T)                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##1 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##2 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##3 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##4 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }
SLANG_VECTOR_GET_ELEMENT(int)
SLANG_VECTOR_GET_ELEMENT(uint)
SLANG_VECTOR_GET_ELEMENT(short)
SLANG_VECTOR_GET_ELEMENT(ushort)
SLANG_VECTOR_GET_ELEMENT(char)
SLANG_VECTOR_GET_ELEMENT(uchar)
SLANG_VECTOR_GET_ELEMENT(longlong)
SLANG_VECTOR_GET_ELEMENT(ulonglong)
SLANG_VECTOR_GET_ELEMENT(float)
SLANG_VECTOR_GET_ELEMENT(double)

#define SLANG_VECTOR_GET_ELEMENT_PTR(T)                                                      \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##1 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##2 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##3 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##4 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }
SLANG_VECTOR_GET_ELEMENT_PTR(int)
SLANG_VECTOR_GET_ELEMENT_PTR(uint)
SLANG_VECTOR_GET_ELEMENT_PTR(short)
SLANG_VECTOR_GET_ELEMENT_PTR(ushort)
SLANG_VECTOR_GET_ELEMENT_PTR(char)
SLANG_VECTOR_GET_ELEMENT_PTR(uchar)
SLANG_VECTOR_GET_ELEMENT_PTR(longlong)
SLANG_VECTOR_GET_ELEMENT_PTR(ulonglong)
SLANG_VECTOR_GET_ELEMENT_PTR(float)
SLANG_VECTOR_GET_ELEMENT_PTR(double)

#if SLANG_CUDA_ENABLE_HALF
SLANG_VECTOR_GET_ELEMENT(__half)
SLANG_VECTOR_GET_ELEMENT_PTR(__half)
#endif

#define SLANG_CUDA_VECTOR_BINARY_OP(T, n, op)                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal, T##n other)             \
    {                                                                                         \
        T##n result;                                                                          \
        for (int i = 0; i < n; i++)                                                           \
            *_slang_vector_get_element_ptr(&result, i) =                                      \
                _slang_vector_get_element(thisVal, i) op _slang_vector_get_element(other, i); \
        return result;                                                                        \
    }
#define SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, op)                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool##n operator op(T##n thisVal, T##n other) \
    {                                                                                \
        bool##n result;                                                              \
        for (int i = 0; i < n; i++)                                                  \
            *_slang_vector_get_element_ptr(&result, i) =                             \
                (int)(_slang_vector_get_element(thisVal, i)                          \
                          op _slang_vector_get_element(other, i));                   \
        return result;                                                               \
    }
#define SLANG_CUDA_VECTOR_UNARY_OP(T, n, op)                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal)                              \
    {                                                                                              \
        T##n result;                                                                               \
        for (int i = 0; i < n; i++)                                                                \
            *_slang_vector_get_element_ptr(&result, i) = op _slang_vector_get_element(thisVal, i); \
        return result;                                                                             \
    }

#define SLANG_CUDA_VECTOR_INT_OP(T, n)            \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, %)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ^)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, |)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, >>)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, <<)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, !)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, ~)

#define SLANG_CUDA_VECTOR_INT_OPS(T) \
    SLANG_CUDA_VECTOR_INT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 4)

SLANG_CUDA_VECTOR_INT_OPS(int)
SLANG_CUDA_VECTOR_INT_OPS(uint)
SLANG_CUDA_VECTOR_INT_OPS(ushort)
SLANG_CUDA_VECTOR_INT_OPS(short)
SLANG_CUDA_VECTOR_INT_OPS(char)
SLANG_CUDA_VECTOR_INT_OPS(uchar)
SLANG_CUDA_VECTOR_INT_OPS(longlong)
SLANG_CUDA_VECTOR_INT_OPS(ulonglong)

#define SLANG_CUDA_VECTOR_FLOAT_OP(T, n)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)
#define SLANG_CUDA_VECTOR_FLOAT_OPS(T) \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 4)

SLANG_CUDA_VECTOR_FLOAT_OPS(float)
SLANG_CUDA_VECTOR_FLOAT_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_CUDA_VECTOR_FLOAT_OPS(__half)
#endif
#define SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, n)                                             \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator%(const T##n& left, const T##n& right) \
    {                                                                                      \
        T##n result;                                                                       \
        for (int i = 0; i < n; i++)                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_fmod(                      \
                _slang_vector_get_element(left, i),                                        \
                _slang_vector_get_element(right, i));                                      \
        return result;                                                                     \
    }
#define SLANG_CUDA_FLOAT_VECTOR_MOD(T)     \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 2) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 3) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 4)

SLANG_CUDA_FLOAT_VECTOR_MOD(float)
SLANG_CUDA_FLOAT_VECTOR_MOD(double)

#if SLANG_CUDA_RTC || SLANG_CUDA_ENABLE_HALF
#define SLANG_MAKE_VECTOR(T)                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x, T y)           \
    {                                                                       \
        return T##2 {x, y};                                                 \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x, T y, T z)      \
    {                                                                       \
        return T##3 {x, y, z};                                              \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x, T y, T z, T w) \
    {                                                                       \
        return T##4 {x, y, z, w};                                           \
    }
#endif

#if SLANG_CUDA_RTC
SLANG_MAKE_VECTOR(int)
SLANG_MAKE_VECTOR(uint)
SLANG_MAKE_VECTOR(short)
SLANG_MAKE_VECTOR(ushort)
SLANG_MAKE_VECTOR(char)
SLANG_MAKE_VECTOR(uchar)
SLANG_MAKE_VECTOR(float)
SLANG_MAKE_VECTOR(double)
SLANG_MAKE_VECTOR(longlong)
SLANG_MAKE_VECTOR(ulonglong)
#endif

#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR(__half)
#endif

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool1 make_bool1(bool x)
{
    return bool1{x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x, bool y)
{
    return bool2{x, y};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x, bool y, bool z)
{
    return bool3{x, y, z};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x, bool y, bool z, bool w)
{
    return bool4{x, y, z, w};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x)
{
    return bool2{x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x)
{
    return bool3{x, x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x)
{
    return bool4{x, x, x, x};
}

#if SLANG_CUDA_RTC
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##1 make_##T##1(T x) \
    {                                                        \
        return T##1 {x};                                     \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#else
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#endif
SLANG_MAKE_VECTOR_FROM_SCALAR(int)
SLANG_MAKE_VECTOR_FROM_SCALAR(uint)
SLANG_MAKE_VECTOR_FROM_SCALAR(short)
SLANG_MAKE_VECTOR_FROM_SCALAR(ushort)
SLANG_MAKE_VECTOR_FROM_SCALAR(char)
SLANG_MAKE_VECTOR_FROM_SCALAR(uchar)
SLANG_MAKE_VECTOR_FROM_SCALAR(longlong)
SLANG_MAKE_VECTOR_FROM_SCALAR(ulonglong)
SLANG_MAKE_VECTOR_FROM_SCALAR(float)
SLANG_MAKE_VECTOR_FROM_SCALAR(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR_FROM_SCALAR(__half)
#if !SLANG_CUDA_RTC
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half1 make___half1(__half x)
{
    return __half1{x};
}
#endif
#endif

#define SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(Fn, T, N)                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##N Fn(T##N* address, T##N val)                           \
    {                                                                                             \
        T##N result;                                                                              \
        for (int i = 0; i < N; i++)                                                               \
            *_slang_vector_get_element_ptr(&result, i) =                                          \
                Fn(_slang_vector_get_element_ptr(address, i), _slang_vector_get_element(val, i)); \
        return result;                                                                            \
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 4)
#endif
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 4)

template<typename T, int n>
struct GetVectorTypeImpl
{
};

#define GET_VECTOR_TYPE_IMPL(T, n)                                     \
    template<>                                                         \
    struct GetVectorTypeImpl<T, n>                                     \
    {                                                                  \
        typedef T##n type;                                             \
        static SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n fromScalar(T v) \
        {                                                              \
            return make_##T##n(v);                                     \
        }                                                              \
    };
#define GET_VECTOR_TYPE_IMPL_N(T) \
    GET_VECTOR_TYPE_IMPL(T, 1)    \
    GET_VECTOR_TYPE_IMPL(T, 2)    \
    GET_VECTOR_TYPE_IMPL(T, 3)    \
    GET_VECTOR_TYPE_IMPL(T, 4)

GET_VECTOR_TYPE_IMPL_N(int)
GET_VECTOR_TYPE_IMPL_N(uint)
GET_VECTOR_TYPE_IMPL_N(short)
GET_VECTOR_TYPE_IMPL_N(ushort)
GET_VECTOR_TYPE_IMPL_N(char)
GET_VECTOR_TYPE_IMPL_N(uchar)
GET_VECTOR_TYPE_IMPL_N(longlong)
GET_VECTOR_TYPE_IMPL_N(ulonglong)
GET_VECTOR_TYPE_IMPL_N(float)
GET_VECTOR_TYPE_IMPL_N(double)
#if SLANG_CUDA_ENABLE_HALF
GET_VECTOR_TYPE_IMPL_N(__half)
#endif
template<typename T, int n>
using Vector = typename GetVectorTypeImpl<T, n>::type;

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, COLS>& operator[](size_t index)
    {
        return rows[index];
    }
};


template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T scalar)
{
    Matrix<T, ROWS, COLS> result;
    for (int i = 0; i < ROWS; i++)
        result.rows[i] = GetVectorTypeImpl<T, COLS>::fromScalar(scalar);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2,
    const Vector<T, COLS>& row3)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    result.rows[3] = row3;
    return result;
}

template<typename T, int ROWS, int COLS, typename U, int otherRow, int otherCol>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Matrix<U, otherRow, otherCol>& other)
{
    Matrix<T, ROWS, COLS> result;
    int minRow = ROWS;
    int minCol = COLS;
    if (minRow > otherRow)
        minRow = otherRow;
    if (minCol > otherCol)
        minCol = otherCol;
    for (int i = 0; i < minRow; i++)
        for (int j = 0; j < minCol; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) =
                (T)_slang_vector_get_element(other.rows[i], j);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[1].x = v2;
    rs.rows[1].y = v3;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 3)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v5;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
        rs.rows[3].x = v6;
        rs.rows[3].y = v7;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[1].x = v3;
    rs.rows[1].y = v4;
    rs.rows[1].z = v5;
    rs.rows[2].x = v6;
    rs.rows[2].y = v7;
    rs.rows[2].z = v8;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
        *_slang_vector_get_element_ptr(&rs.rows[2], 0) = v8;
        *_slang_vector_get_element_ptr(&rs.rows[2], 1) = v9;
        *_slang_vector_get_element_ptr(&rs.rows[2], 2) = v10;
        *_slang_vector_get_element_ptr(&rs.rows[2], 3) = v11;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[0].z = v2;
        rs.rows[1].x = v3;
        rs.rows[1].y = v4;
        rs.rows[1].z = v5;
        rs.rows[2].x = v6;
        rs.rows[2].y = v7;
        rs.rows[2].z = v8;
        rs.rows[3].x = v9;
        rs.rows[3].y = v10;
        rs.rows[3].z = v11;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11,
    T v12,
    T v13,
    T v14,
    T v15)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[0].w = v3;
    rs.rows[1].x = v4;
    rs.rows[1].y = v5;
    rs.rows[1].z = v6;
    rs.rows[1].w = v7;
    rs.rows[2].x = v8;
    rs.rows[2].y = v9;
    rs.rows[2].z = v10;
    rs.rows[2].w = v11;
    rs.rows[3].x = v12;
    rs.rows[3].y = v13;
    rs.rows[3].z = v14;
    rs.rows[3].w = v15;
    return rs;
}

#define SLANG_MATRIX_BINARY_OP(T, op)                                   \
    template<int R, int C>                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(     \
        const Matrix<T, R, C>& thisVal,                                 \
        const Matrix<T, R, C>& other)                                   \
    {                                                                   \
        Matrix<T, R, C> result;                                         \
        for (int i = 0; i < R; i++)                                     \
            for (int j = 0; j < C; j++)                                 \
                *_slang_vector_get_element_ptr(result.rows + i, j) =    \
                    _slang_vector_get_element(thisVal.rows[i], j)       \
                        op _slang_vector_get_element(other.rows[i], j); \
        return result;                                                  \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                                                               \
    template<int R, int C>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                                                              \
        Matrix<T, R, C> result;                                                                    \
        for (int i = 0; i < R; i++)                                                                \
            for (int j = 0; j < C; j++)                                                            \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                               \
                    op _slang_vector_get_element(thisVal.rows[i], j);                              \
        return result;                                                                             \
    }
#define SLANG_INT_MATRIX_OPS(T)   \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_BINARY_OP(T, &)  \
    SLANG_MATRIX_BINARY_OP(T, |)  \
    SLANG_MATRIX_BINARY_OP(T, &&) \
    SLANG_MATRIX_BINARY_OP(T, ||) \
    SLANG_MATRIX_BINARY_OP(T, ^)  \
    SLANG_MATRIX_BINARY_OP(T, %)  \
    SLANG_MATRIX_UNARY_OP(T, !)   \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_UNARY_OP(T, -)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(short)
SLANG_INT_MATRIX_OPS(ushort)
SLANG_INT_MATRIX_OPS(char)
SLANG_INT_MATRIX_OPS(uchar)
SLANG_INT_MATRIX_OPS(longlong)
SLANG_INT_MATRIX_OPS(ulonglong)
SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_FLOAT_MATRIX_OPS(__half)
#endif
#define SLANG_MATRIX_INT_NEG_OP(T)                                                        \
    template<int R, int C>                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                                     \
        Matrix<T, R, C> result;                                                           \
        for (int i = 0; i < R; i++)                                                       \
            for (int j = 0; j < C; j++)                                                   \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                      \
                    0 - _slang_vector_get_element(thisVal.rows[i], j);                    \
        return result;                                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(short)
SLANG_MATRIX_INT_NEG_OP(ushort)
SLANG_MATRIX_INT_NEG_OP(char)
SLANG_MATRIX_INT_NEG_OP(uchar)
SLANG_MATRIX_INT_NEG_OP(longlong)
SLANG_MATRIX_INT_NEG_OP(ulonglong)

#define SLANG_FLOAT_MATRIX_MOD(T)                                                 \
    template<int R, int C>                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator%(                 \
        Matrix<T, R, C> left,                                                     \
        Matrix<T, R, C> right)                                                    \
    {                                                                             \
        Matrix<T, R, C> result;                                                   \
        for (int i = 0; i < R; i++)                                               \
            for (int j = 0; j < C; j++)                                           \
                *_slang_vector_get_element_ptr(result.rows + i, j) = _slang_fmod( \
                    _slang_vector_get_element(left.rows[i], j),                   \
                    _slang_vector_get_element(right.rows[i], j));                 \
        return result;                                                            \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#if SLANG_CUDA_ENABLE_HALF
template<int R, int C>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<__half, R, C> operator%(
    Matrix<__half, R, C> left,
    Matrix<__half, R, C> right)
{
    Matrix<__half, R, C> result;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) = __float2half(_slang_fmod(
                __half2float(_slang_vector_get_element(left.rows[i], j)),
                __half2float(_slang_vector_get_element(right.rows[i], j))));
    return result;
}
#endif
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

#define SLANG_SELECT_IMPL(T, N)                                                                  \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, N> _slang_select(                               \
        bool##N condition,                                                                       \
        Vector<T, N> v0,                                                                         \
        Vector<T, N> v1)                                                                         \
    {                                                                                            \
        Vector<T, N> result;                                                                     \
        for (int i = 0; i < N; i++)                                                              \
        {                                                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_vector_get_element(condition, i) \
                                                             ? _slang_vector_get_element(v0, i)  \
                                                             : _slang_vector_get_element(v1, i); \
        }                                                                                        \
        return result;                                                                           \
    }
#define SLANG_SELECT_T(T)   \
    SLANG_SELECT_IMPL(T, 2) \
    SLANG_SELECT_IMPL(T, 3) \
    SLANG_SELECT_IMPL(T, 4)

SLANG_SELECT_T(int)
SLANG_SELECT_T(uint)
SLANG_SELECT_T(short)
SLANG_SELECT_T(ushort)
SLANG_SELECT_T(char)
SLANG_SELECT_T(uchar)
SLANG_SELECT_T(float)
SLANG_SELECT_T(double)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

//
// Half support
//

#if SLANG_CUDA_ENABLE_HALF
SLANG_SELECT_T(__half)

// Convenience functions ushort -> half

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 __ushort_as_half(const ushort2& i)
{
    return __halves2half2(__ushort_as_half(i.x), __ushort_as_half(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half3 __ushort_as_half(const ushort3& i)
{
    return __half3{__ushort_as_half(i.x), __ushort_as_half(i.y), __ushort_as_half(i.z)};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 __ushort_as_half(const ushort4& i)
{
    return __half4{
        __ushort_as_half(i.x),
        __ushort_as_half(i.y),
        __ushort_as_half(i.z),
        __ushort_as_half(i.w)};
}

// Convenience functions half -> ushort

SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort2 __half_as_ushort(const __half2& i)
{
    return make_ushort2(__half_as_ushort(i.x), __half_as_ushort(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort3 __half_as_ushort(const __half3& i)
{
    return make_ushort3(__half_as_ushort(i.x), __half_as_ushort(i.y), __half_as_ushort(i.z));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort4 __half_as_ushort(const __half4& i)
{
    return make_ushort4(
        __half_as_ushort(i.x),
        __half_as_ushort(i.y),
        __half_as_ushort(i.z),
        __half_as_ushort(i.w));
}

// This is a little bit of a hack. Fortunately CUDA has the definitions of the templated types in
// include/surface_indirect_functions.h
// Here we find the template definition requires a specialization of __nv_isurf_trait to allow
// a specialization of the surface write functions.
// This *isn't* a problem on the read functions as they don't have a return type that uses this
// mechanism

template<>
struct __nv_isurf_trait<__half>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half2>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half4>
{
    typedef void type;
};

#define SLANG_DROP_PARENS(...) __VA_ARGS__

#define SLANG_SURFACE_READ(FUNC_NAME, TYPE_ARGS, ARGS)                                             \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half FUNC_NAME<__half>(                                   \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(FUNC_NAME<ushort>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 FUNC_NAME<__half2>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 FUNC_NAME<__half4>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }

SLANG_SURFACE_READ(surf1Dread, (int x), (x))
SLANG_SURFACE_READ(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ(surf3Dread, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_READ(surf1DLayeredread, (int x, int layer), (x, layer))
SLANG_SURFACE_READ(surf2DLayeredread, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_READ(surfCubemapread, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_READ(surfCubemapLayeredread, (int x, int y, int layerFace), (x, y, layerFace))

#define SLANG_SURFACE_WRITE(FUNC_NAME, TYPE_ARGS, ARGS)                                            \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half>(                                     \
        __half data,                                                                               \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode);  \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half2>(                                    \
        __half2 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort2>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half4>(                                    \
        __half4 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort4>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }

SLANG_SURFACE_WRITE(surf1Dwrite, (int x), (x))
SLANG_SURFACE_WRITE(surf2Dwrite, (int x, int y), (x, y))
SLANG_SURFACE_WRITE(surf3Dwrite, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_WRITE(surf1DLayeredwrite, (int x, int layer), (x, layer))
SLANG_SURFACE_WRITE(surf2DLayeredwrite, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_WRITE(surfCubemapwrite, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_WRITE(surfCubemapLayeredwrite, (int x, int y, int layerFace), (x, y, layerFace))

// ! Hack to test out reading !!!
// Only works converting *from* half

// template <typename T>
// SLANG_FORCE_INLINE SLANG_CUDA_CALL T surf2Dread_convert(cudaSurfaceObject_t surfObj, int x, int
// y, cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURFACE_READ_HALF_CONVERT(FUNC_NAME, TYPE_ARGS, ARGS)                              \
                                                                                                 \
    template<typename T>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T FUNC_NAME##_convert(                                    \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode);                                                   \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float FUNC_NAME##_convert<float>(                         \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        return __ushort_as_half(                                                                 \
            FUNC_NAME<uint16_t>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float2 FUNC_NAME##_convert<float2>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half2 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float2{v.x, v.y};                                                                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float4 FUNC_NAME##_convert<float4>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half4 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float4{v.x, v.y, v.z, v.w};                                                       \
    }

SLANG_SURFACE_READ_HALF_CONVERT(surf1Dread, (int x), (x))
SLANG_SURFACE_READ_HALF_CONVERT(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ_HALF_CONVERT(surf3Dread, (int x, int y, int z), (x, y, z))

#endif

// Support for doing format conversion when writing to a surface/RWTexture

// NOTE! For normal surface access x values are *byte* addressed.
// For the _convert versions they are *not*. They don't need to be because sust.p does not require
// it.

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode);

// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust

// Float

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(v));
}

// Float2

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3,%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy));
}

// Float4
template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3,%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2}], {%3,%4,%5,%6};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2,%3}], {%4,%5,%6,%7};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

// ----------------------------- F32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_ceil(float f)
{
    return ::ceilf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_floor(float f)
{
    return ::floorf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_round(float f)
{
    return ::roundf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sin(float f)
{
    return ::sinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cos(float f)
{
    return ::cosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F32_sincos(float f, float* s, float* c)
{
    ::sincosf(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tan(float f)
{
    return ::tanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_asin(float f)
{
    return ::asinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_acos(float f)
{
    return ::acosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan(float f)
{
    return ::atanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sinh(float f)
{
    return ::sinhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cosh(float f)
{
    return ::coshf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tanh(float f)
{
    return ::tanhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log2(float f)
{
    return ::log2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log(float f)
{
    return ::logf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log10(float f)
{
    return ::log10f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp2(float f)
{
    return ::exp2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp(float f)
{
    return ::expf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_abs(float f)
{
    return ::fabsf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_trunc(float f)
{
    return ::truncf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sqrt(float f)
{
    return ::sqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_rsqrt(float f)
{
    return ::rsqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sign(float f)
{
    return (f == 0.0f) ? f : ((f < 0.0f) ? -1.0f : 1.0f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frac(float f)
{
    return f - F32_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isnan(float f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isfinite(float f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isinf(float f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_min(float a, float b)
{
    return ::fminf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_max(float a, float b)
{
    return ::fmaxf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_pow(float a, float b)
{
    return ::powf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fmod(float a, float b)
{
    return ::fmodf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_remainder(float a, float b)
{
    return ::remainderf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan2(float a, float b)
{
    return float(::atan2(a, b));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frexp(float x, int* e)
{
    return frexpf(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t F32_asuint(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t F32_asint(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fma(float a, float b, float c)
{
    return ::fmaf(a, b, c);
}


// ----------------------------- F64 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_ceil(double f)
{
    return ::ceil(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_floor(double f)
{
    return ::floor(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_round(double f)
{
    return ::round(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sin(double f)
{
    return ::sin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cos(double f)
{
    return ::cos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_sincos(double f, double* s, double* c)
{
    ::sincos(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tan(double f)
{
    return ::tan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_asin(double f)
{
    return ::asin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_acos(double f)
{
    return ::acos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan(double f)
{
    return ::atan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sinh(double f)
{
    return ::sinh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cosh(double f)
{
    return ::cosh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tanh(double f)
{
    return ::tanh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log2(double f)
{
    return ::log2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log(double f)
{
    return ::log(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log10(float f)
{
    return ::log10(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp2(double f)
{
    return ::exp2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp(double f)
{
    return ::exp(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_abs(double f)
{
    return ::fabs(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_trunc(double f)
{
    return ::trunc(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sqrt(double f)
{
    return ::sqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_rsqrt(double f)
{
    return ::rsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sign(double f)
{
    return (f == 0.0) ? f : ((f < 0.0) ? -1.0 : 1.0);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frac(double f)
{
    return f - F64_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isnan(double f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isfinite(double f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isinf(double f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_min(double a, double b)
{
    return ::fmin(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_max(double a, double b)
{
    return ::fmax(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_pow(double a, double b)
{
    return ::pow(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fmod(double a, double b)
{
    return ::fmod(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_remainder(double a, double b)
{
    return ::remainder(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan2(double a, double b)
{
    return ::atan2(a, b);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frexp(double x, int* e)
{
    return ::frexp(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

// ----------------------------- I32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_abs(int32_t f)
{
    return (f < 0) ? -f : f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float I32_asfloat(int32_t x)
{
    Union32 u;
    u.i = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_asuint(int32_t x)
{
    return uint32_t(x);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double I32_asdouble(int32_t low, int32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

// ----------------------------- U32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_abs(uint32_t f)
{
    return f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float U32_asfloat(uint32_t x)
{
    Union32 u;
    u.u = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_asint(int32_t x)
{
    return uint32_t(x);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_countbits(uint32_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popc(v);
}


// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_abs(int64_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_abs(uint64_t f)
{
    return f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U64_countbits(uint64_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popcll(v);
}


// ----------------------------- ResourceType -----------------------------------------


// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct StructuredBuffer
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

    SLANG_CUDA_CALL const T& Load(size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }
#endif

    T* data;
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    size_t count;
#endif
};

template<typename T>
struct RWStructuredBuffer : StructuredBuffer<T>
{
    SLANG_CUDA_CALL T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, this->count);
#endif
        return this->data[index];
    }
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }
    template<typename T>
    SLANG_CUDA_CALL StructuredBuffer<T> asStructuredBuffer() const
    {
        StructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Missing support for Atomic operations
// Missing support for Load with status
struct RWByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }

    SLANG_CUDA_CALL void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    SLANG_CUDA_CALL void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    SLANG_CUDA_CALL void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    SLANG_CUDA_CALL void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    SLANG_CUDA_CALL void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        memcpy((char*)data + index, &value, sizeof(T));
    }

    /// Can be used in the core module to gain access
    template<typename T>
    SLANG_CUDA_CALL T* _getPtrAt(size_t index)
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return (T*)(((char*)data) + index);
    }
    template<typename T>
    SLANG_CUDA_CALL RWStructuredBuffer<T> asStructuredBuffer() const
    {
        RWStructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};


// ---------------------- Wave --------------------------------------

// TODO(JS): It appears that cuda does not have a simple way to get a lane index.
//
// Another approach could be...
// laneId = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) &
// SLANG_CUDA_WARP_MASK If that is really true another way to do this, would be for code generator
// to add this function with the [numthreads] baked in.
//
// For now I'll just assume you have a launch that makes the following correct if the kernel uses
// WaveGetLaneIndex()
#ifndef SLANG_USE_ASM_LANE_ID
__forceinline__ __device__ uint32_t _getLaneId()
{
    // If the launch is (or I guess some multiple of the warp size)
    // we try this mechanism, which is apparently faster.
    return threadIdx.x & SLANG_CUDA_WARP_MASK;
}
#else
__forceinline__ __device__ uint32_t _getLaneId()
{
    // https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid#
    // This mechanism is not the fastest way to do it, and that is why the other mechanism
    // is the default. But the other mechanism relies on a launch that makes the assumption
    // true.
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
#endif

typedef int WarpMask;

// It appears that the __activemask() cannot always be used because
// threads need to be converged.
//
// For CUDA the article claims mask has to be used carefully
// https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
// With the Warp intrinsics there is no mask, and it's just the 'active lanes'.
// __activemask() though does not require there is convergence, so that doesn't work.
//
// '__ballot_sync' produces a convergance.
//
// From the CUDA docs:
// ```For __all_sync, __any_sync, and __ballot_sync, a mask must be passed that specifies the
// threads participating in the call. A bit, representing the thread's lane ID, must be set for each
// participating thread to ensure they are properly converged before the intrinsic is executed by
// the hardware. All active threads named in mask must execute the same intrinsic with the same
// mask, or the result is undefined.```
//
// Currently there isn't a mechanism to correctly get the mask without it being passed through.
// Doing so will most likely require some changes to slang code generation to track masks, for now
// then we use _getActiveMask.

// Return mask of all the lanes less than the current lane
__forceinline__ __device__ WarpMask _getLaneLtMask()
{
    return (int(1) << _getLaneId()) - 1;
}

// TODO(JS):
// THIS IS NOT CORRECT! That determining the appropriate active mask requires appropriate
// mask tracking.
__forceinline__ __device__ WarpMask _getActiveMask()
{
    return __ballot_sync(__activemask(), true);
}

// Return a mask suitable for the 'MultiPrefix' style functions
__forceinline__ __device__ WarpMask _getMultiPrefixMask(int mask)
{
    return mask;
}

// Note! Note will return true if mask is 0, but thats okay, because there must be one
// lane active to execute anything
__inline__ __device__ bool _waveIsSingleLane(WarpMask mask)
{
    return (mask & (mask - 1)) == 0;
}

// Returns the power of 2 size of run of set bits. Returns 0 if not a suitable run.
// Examples:
// 0b00000000'00000000'00000000'11111111 -> 8
// 0b11111111'11111111'11111111'11111111 -> 32
// 0b00000000'00000000'00000000'00011111 -> 0 (since 5 is not a power of 2)
// 0b00000000'00000000'00000000'11110000 -> 0 (since the run of bits does not start at the LSB)
// 0b00000000'00000000'00000000'00100111 -> 0 (since it is not a single contiguous run)
__inline__ __device__ int _waveCalcPow2Offset(WarpMask mask)
{
    // This should be the most common case, so fast path it
    if (mask == SLANG_CUDA_WARP_BITMASK)
    {
        return SLANG_CUDA_WARP_SIZE;
    }
    // Is it a contiguous run of bits?
    if ((mask & (mask + 1)) == 0)
    {
        // const int offsetSize = __ffs(mask + 1) - 1;
        const int offset = 32 - __clz(mask);
        // Is it a power of 2 size
        if ((offset & (offset - 1)) == 0)
        {
            return offset;
        }
    }
    return 0;
}

__inline__ __device__ bool _waveIsFirstLane()
{
    const WarpMask mask = __activemask();
    // We special case bit 0, as that most warps are expected to be fully active.

    // mask & -mask, isolates the lowest set bit.
    // return (mask & 1 ) || ((mask & -mask) == (1 << _getLaneId()));

    // This mechanism is most similar to what was in an nVidia post, so assume it is prefered.
    return (mask & 1) || ((__ffs(mask) - 1) == _getLaneId());
}

template<typename T>
struct WaveOpOr
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a | b; }
};

template<typename T>
struct WaveOpAnd
{
    __inline__ __device__ static T getInitial(T a) { return ~T(0); }
    __inline__ __device__ static T doOp(T a, T b) { return a & b; }
};

template<typename T>
struct WaveOpXor
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a ^ b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a ^ b; }
};

template<typename T>
struct WaveOpAdd
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a + b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a - b; }
};

template<typename T>
struct WaveOpMul
{
    __inline__ __device__ static T getInitial(T a) { return T(1); }
    __inline__ __device__ static T doOp(T a, T b) { return a * b; }
    // Using this inverse for int is probably undesirable - because in general it requires T to have
    // more precision There is also a performance aspect to it, where divides are generally
    // significantly slower
    __inline__ __device__ static T doInverse(T a, T b) { return a / b; }
};

template<typename T>
struct WaveOpMax
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a > b ? a : b; }
};

template<typename T>
struct WaveOpMin
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a < b ? a : b; }
};

template<typename T>
struct ElementTypeTrait;

// Scalar
template<>
struct ElementTypeTrait<int>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<uint>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<float>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<double>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<uint64_t>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<int64_t>
{
    typedef int64_t Type;
};

// Vector
template<>
struct ElementTypeTrait<int1>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int2>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int3>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int4>
{
    typedef int Type;
};

template<>
struct ElementTypeTrait<uint1>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint2>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint3>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint4>
{
    typedef uint Type;
};

template<>
struct ElementTypeTrait<float1>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float2>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float3>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float4>
{
    typedef float Type;
};

template<>
struct ElementTypeTrait<double1>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double2>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double3>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double4>
{
    typedef double Type;
};

// Matrix
template<typename T, int ROWS, int COLS>
struct ElementTypeTrait<Matrix<T, ROWS, COLS>>
{
    typedef T Type;
};

// Scalar
template<typename INTF, typename T>
__device__ T _waveReduceScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            val = INTF::doOp(val, __shfl_xor_sync(mask, val, offset));
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        T result = INTF::getInitial(val);
        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            result = INTF::doOp(result, __shfl_sync(mask, val, srcLane));
            remaining &= ~laneBit;
        }
        return result;
    }
    return val;
}


// Multiple values
template<typename INTF, typename T, size_t COUNT>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_xor_sync(mask, val[i], offset));
            }
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        // Copy the original
        T originalVal[COUNT];
        for (size_t i = 0; i < COUNT; ++i)
        {
            const T v = val[i];
            originalVal[i] = v;
            val[i] = INTF::getInitial(v);
        }

        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_sync(mask, originalVal[i], srcLane));
            }
            remaining &= ~laneBit;
        }
    }
}

template<typename INTF, typename T>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<INTF, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)val);
}

template<typename T>
__inline__ __device__ T _waveOr(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveAnd(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAnd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveXor(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveProduct(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveSum(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMin(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMax(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMax<T>, T>(mask, val);
}

// Fast-path specializations when CUDA warp reduce operators are available
#if __CUDA_ARCH__ >= 800 // 8.x or higher
template<>
__inline__ __device__ unsigned _waveOr<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_or_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveAnd<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_and_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveXor<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_xor_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveSum<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ int _waveSum<int>(WarpMask mask, int val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMin<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMin<int>(WarpMask mask, int val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMax<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_max_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMax<int>(WarpMask mask, int val)
{
    return __reduce_max_sync(mask, val);
}
#endif


// Multiple

template<typename T>
__inline__ __device__ T _waveOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpOr<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAnd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpXor<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMul<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAdd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMin<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMax<ElemType>>(mask, &val);
    return val;
}


template<typename T>
__inline__ __device__ bool _waveAllEqual(WarpMask mask, T val)
{
    int pred;
    __match_all_sync(mask, val, &pred);
    return pred != 0;
}

template<typename T>
__inline__ __device__ bool _waveAllEqualMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    for (size_t i = 0; i < count; ++i)
    {
        __match_all_sync(mask, src[i], &pred);
        if (pred == 0)
        {
            return false;
        }
    }
    return true;
}

template<typename T>
__inline__ __device__ T _waveReadFirst(WarpMask mask, T val)
{
    const int lowestLaneId = __ffs(mask) - 1;
    return __shfl_sync(mask, val, lowestLaneId);
}

template<typename T>
__inline__ __device__ T _waveReadFirstMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    const int lowestLaneId = __ffs(mask) - 1;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lowestLaneId);
    }
    return outVal;
}

template<typename T>
__inline__ __device__ T _waveShuffleMultiple(WarpMask mask, T inVal, int lane)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lane);
    }
    return outVal;
}

// Scalar

// Invertable means that when we get to the end of the reduce, we can remove val (to make
// exclusive), using the inverse of the op.
template<typename INTF, typename T>
__device__ T _wavePrefixInvertableScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result;
    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        result = val;
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, result, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
            }
        }
        // Remove val from the result, by applyin inverse
        result = INTF::doInverse(result, val);
    }
    else
    {
        result = INTF::getInitial(val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


// This implementation separately tracks the value to be propogated, and the value
// that is the final result
template<typename INTF, typename T>
__device__ T _wavePrefixScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result = INTF::getInitial(val);
    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra multiply for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, val, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
                val = INTF::doOp(val, readVal);
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpCopy(T* dst, const T* src)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        dst[j] = src[j];
    }
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpDoInverse(T* inOut, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        inOut[j] = INTF::doInverse(inOut[j], val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpSetInitial(T* out, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        out[j] = INTF::getInitial(val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixInvertableMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T originalVal[COUNT];
    _waveOpCopy<INTF, T, COUNT>(originalVal, val);

    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        for (int i = 1; i < offsetSize; i += i)
        {
            // TODO(JS): Note that here I don't split the laneId outside so it's only tested once.
            // This may be better but it would also mean that there would be shfl between lanes
            // that are on different (albeit identical) instructions. So this seems more likely to
            // work as expected with everything in lock step.
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, val[j], i, offsetSize);
                if (laneId >= i)
                {
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
        // Remove originalVal from the result, by applyin inverse
        _waveOpDoInverse<INTF, T, COUNT>(val, originalVal);
    }
    else
    {
        _waveOpSetInitial<INTF, T, COUNT>(val, val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, originalVal[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                    remaining &= ~laneBit;
                }
            }
        }
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();

    T work[COUNT];
    _waveOpCopy<INTF, T, COUNT>(work, val);
    _waveOpSetInitial<INTF, T, COUNT>(val, val);

    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra op for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, work[j], i, offsetSize);
                if (laneId >= i)
                {
                    work[j] = INTF::doOp(work[j], readVal);
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, work[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                }
                remaining &= ~laneBit;
            }
        }
    }
}

template<typename T>
__inline__ __device__ T _wavePrefixProduct(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixSum(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixXor(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixOr(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixAnd(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpAnd<T>, T>(mask, val);
}


template<typename T>
__inline__ __device__ T _wavePrefixProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpMul<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpAdd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpXor<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpOr<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpAnd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ uint4 _waveMatchScalar(WarpMask mask, T val)
{
    int pred;
    return make_uint4(__match_all_sync(mask, val, &pred), 0, 0, 0);
}

template<typename T>
__inline__ __device__ uint4 _waveMatchMultiple(WarpMask mask, const T& inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    uint matchBits = 0xffffffff;
    for (size_t i = 0; i < count && matchBits; ++i)
    {
        matchBits = matchBits & __match_all_sync(mask, src[i], &pred);
    }
    return make_uint4(matchBits, 0, 0, 0);
}

__device__ uint getAt(dim3 a, int b)
{
    SLANG_PRELUDE_ASSERT(b >= 0 && b < 3);
    return (&a.x)[b];
}
__device__ uint3 operator*(uint3 a, dim3 b)
{
    uint3 r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r;
}

template<typename TResult, typename TInput>
__inline__ __device__ TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */


/* Type that defines the uniform entry point params. The actual content of this type is dependent on
the entry point parameters, and can be found via reflection or defined such that it matches the
shader appropriately.
*/
struct UniformEntryPointParams;
struct UniformState;

// ---------------------- OptiX Ray Payload --------------------------------------
#ifdef SLANG_CUDA_ENABLE_OPTIX
struct RayDesc
{
    float3 Origin;
    float TMin;
    float3 Direction;
    float TMax;
};

static __forceinline__ __device__ void* unpackOptiXRayPayloadPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packOptiXRayPayloadPointer(
    void* ptr,
    uint32_t& i0,
    uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* getOptiXRayPayloadPtr()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpackOptiXRayPayloadPointer(u0, u1);
}

template<typename T>
__forceinline__ __device__ void* traceOptiXRay(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T* Payload)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTrace(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f, /* Time for motion blur, currently unsupported in slang */
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0,
        r1);
}

#endif

static const int kSlangTorchTensorMaxDim = 5;

// TensorView
struct TensorView
{
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;

    template<typename T>
    __device__ T* data_ptr()
    {
        return reinterpret_cast<T*>(data);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint32_t index)
    {
        uint64_t offset = strides[0] * index;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint2 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint3 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint4 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
                          strides[3] * index.w;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T, unsigned int N>
    __device__ T* data_ptr_at(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T& load(uint32_t x)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y);
    }
    template<typename T>
    __device__ T& load(uint2 index)
    {
        return *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z);
    }
    template<typename T>
    __device__ T& load(uint3 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w);
    }
    template<typename T>
    __device__ T& load(uint4 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w);
    }
    template<typename T>
    __device__ T& load(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4);
    }

    // Generic version of load
    template<typename T, unsigned int N>
    __device__ T& load(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return *reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ void store(uint32_t x, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y) = val;
    }
    template<typename T>
    __device__ void store(uint2 index, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z) = val;
    }
    template<typename T>
    __device__ void store(uint3 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, uint32_t w, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w) = val;
    }
    template<typename T>
    __device__ void store(uint4 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w) = val;
    }
    template<typename T>
    __device__ void store(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4) = val;
    }

    // Generic version
    template<typename T, unsigned int N>
    __device__ void store(uint index[N], T val)
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        *reinterpret_cast<T*>(data + offset) = val;
    }
};


#line 2 "./prototyping/bilateral_filter.slang"
__device__ float bilteral_filter_0(int num_features_0, int width_0, int height_0, int b_0, int y_0, int x_0, TensorView input_0, TensorView params_0, TensorView output_0, int kernel_boundary_0, int dialation_0, bool normalize_0)
{

#line 14
    int xoff_0;



    int _S1 = - kernel_boundary_0;

#line 18
    int yoff_0 = _S1;

#line 18
    float total_weight_0 = 0.0f;

#line 29
    int _S2 = num_features_0 + int(2);



    uint _S3 = uint(b_0);

#line 33
    uint _S4 = uint(y_0);

#line 33
    uint _S5 = uint(x_0);

#line 18
    for(;;)
    {

#line 18
        if(yoff_0 <= kernel_boundary_0)
        {
        }
        else
        {

#line 18
            break;
        }

#line 18
        xoff_0 = _S1;

#line 18
        float total_weight_1 = total_weight_0;
        for(;;)
        {

#line 19
            if(xoff_0 <= kernel_boundary_0)
            {
            }
            else
            {

#line 19
                break;
            }

#line 20
            int xp_0 = x_0 + xoff_0 * dialation_0;
            int yp_0 = y_0 + yoff_0 * dialation_0;

#line 21
            bool _S6;


            if(xp_0 < int(0))
            {

#line 24
                _S6 = true;

#line 24
            }
            else
            {

#line 24
                _S6 = yp_0 < int(0);

#line 24
            }

#line 24
            bool _S7;

#line 24
            if(_S6)
            {

#line 24
                _S7 = true;

#line 24
            }
            else
            {

#line 24
                _S7 = xp_0 >= width_0;

#line 24
            }

#line 24
            bool _S8;

#line 24
            if(_S7)
            {

#line 24
                _S8 = true;

#line 24
            }
            else
            {

#line 24
                _S8 = yp_0 >= height_0;

#line 24
            }

#line 24
            if(_S8)
            {

#line 25
                xoff_0 = xoff_0 + int(1);

#line 19
                continue;
            }

#line 19
            int i_0;

#line 19
            int i_1 = int(0);

#line 19
            float sample_weight_0 = 1.0f;

#line 34
            uint _S9 = uint(yp_0);

#line 34
            uint _S10 = uint(xp_0);

#line 29
            for(;;)
            {

#line 29
                if(i_1 < _S2)
                {
                }
                else
                {

#line 29
                    break;
                }

#line 29
                float sample_val_0;

#line 29
                float center_val_0;


                if(i_1 < num_features_0)
                {

#line 33
                    float _S11 = ((input_0).load<float>((_S3), (uint(i_1)), (_S4), (_S5)));
                    float _S12 = ((input_0).load<float>((_S3), (uint(i_1)), (_S9), (_S10)));

#line 34
                    sample_val_0 = _S12;

#line 34
                    center_val_0 = _S11;

#line 32
                }
                else
                {


                    if(i_1 == num_features_0)
                    {

#line 37
                        i_0 = yoff_0;

#line 37
                    }
                    else
                    {

#line 37
                        i_0 = xoff_0;

#line 37
                    }

#line 37
                    sample_val_0 = float(i_0);

#line 37
                    center_val_0 = 0.0f;

#line 32
                }

#line 41
                float diff_0 = sample_val_0 - center_val_0;

#line 49
                float _S13 = ((params_0).load<float>((uint(i_1))));

                float sample_weight_1 = sample_weight_0 * (F32_exp((_S13 * diff_0 * diff_0)));

#line 29
                i_1 = i_1 + int(1);

#line 29
                sample_weight_0 = sample_weight_1;

#line 29
            }

#line 29
            i_0 = int(0);

#line 54
            for(;;)
            {

#line 54
                if(i_0 < num_features_0)
                {
                }
                else
                {

#line 54
                    break;
                }

#line 55
                float _S14 = ((input_0).load<float>((_S3), (uint(i_0)), (_S9), (_S10)));
                uint _S15 = uint(i_0);

#line 56
                float _S16 = ((output_0).load<float>((_S3), (_S15), (_S4), (_S5)));

#line 56
                (output_0).store<float>((_S3), (_S15), (_S4), (_S5), (_S16 + sample_weight_0 * _S14));

#line 54
                i_0 = i_0 + int(1);

#line 54
            }

#line 54
            total_weight_1 = total_weight_1 + sample_weight_0;

#line 19
            xoff_0 = xoff_0 + int(1);

#line 19
        }

#line 18
        yoff_0 = yoff_0 + int(1);

#line 18
        total_weight_0 = total_weight_1;

#line 18
    }

#line 63
    if(total_weight_0 < 0.00100000004749745f)
    {

#line 63
        total_weight_0 = 0.00100000004749745f;

#line 63
    }

    if(normalize_0)
    {

#line 65
        xoff_0 = int(0);
        for(;;)
        {

#line 66
            if(xoff_0 < num_features_0)
            {
            }
            else
            {

#line 66
                break;
            }

#line 67
            uint _S17 = uint(xoff_0);

#line 67
            float _S18 = ((output_0).load<float>((_S3), (_S17), (_S4), (_S5)));

#line 67
            (output_0).store<float>((_S3), (_S17), (_S4), (_S5), (_S18 / total_weight_0));

#line 66
            xoff_0 = xoff_0 + int(1);

#line 66
        }

#line 65
    }

#line 72
    return total_weight_0;
}


#line 619
__global__ void __kernel__exec_bilateral_filter_wrapper(TensorView input_1, TensorView params_1, TensorView output_1, int kernel_boundary_1, int dialation_1)
{

#line 605
    int globalIdx_0 = int(((threadIdx)).x + ((blockIdx)).x * ((blockDim)).x);
    uint _S19 = ((input_1).sizes[(0U)]);

#line 606
    int batch_size_0 = int(_S19);
    uint _S20 = ((input_1).sizes[(1U)]);

#line 607
    int num_features_1 = int(_S20);
    uint _S21 = ((input_1).sizes[(3U)]);

#line 608
    int width_1 = int(_S21);
    uint _S22 = ((input_1).sizes[(2U)]);

#line 609
    int height_1 = int(_S22);
    if(globalIdx_0 >= batch_size_0 * width_1 * height_1)
    {

#line 611
        return;
    }
    int x_1 = globalIdx_0 % width_1;
    int _S23 = globalIdx_0 / width_1;

#line 614
    int y_1 = _S23 % height_1;
    int b_1 = globalIdx_0 / (width_1 * height_1);

#line 628
    float _S24 = bilteral_filter_0(num_features_1, width_1, height_1, b_1, y_1, x_1, input_1, params_1, output_1, kernel_boundary_1, dialation_1, true);
    return;
}


#line 77
__device__ __shared__ FixedArray<FixedArray<float, 256> , 32>  shr_dLdParam_0;

__device__ void bwd_bilteral_filter_0(int num_features_2, int width_2, int height_2, int b_2, int y_2, int x_2, TensorView input_2, TensorView input_grad_0, TensorView params_2, TensorView params_grad_0, TensorView output_2, TensorView output_grad_0, int kernel_boundary_2, int dialation_2)
{

#line 95
    uint _S25 = ((threadIdx)).x;

#line 95
    int _S26 = int(_S25 % 32U);

#line 95
    int i_2;

    if(_S25 < 32U)
    {

#line 97
        i_2 = int(0);
        int _S27 = num_features_2 + int(2);

#line 98
        for(;;)
        {

#line 98
            if(i_2 < _S27)
            {
            }
            else
            {

#line 98
                break;
            }

#line 99
            (*&shr_dLdParam_0)[_S25][i_2] = 0.0f;

#line 98
            i_2 = i_2 + int(1);

#line 98
        }

#line 97
    }

#line 104
    float _S28 = bilteral_filter_0(num_features_2, width_2, height_2, b_2, y_2, x_2, input_2, params_2, output_2, kernel_boundary_2, dialation_2, false);

#line 104
    i_2 = int(0);

#line 104
    float dLdTotalWeight_0 = 0.0f;



    uint _S29 = uint(b_2);

#line 108
    uint _S30 = uint(y_2);

#line 108
    uint _S31 = uint(x_2);

#line 108
    float _S32 = _S28 * _S28;


    int _S33 = - kernel_boundary_2;

#line 178
    int j_0 = int(_S25);
    int _S34 = num_features_2 + int(2);

#line 179
    bool _S35 = j_0 < _S34;

#line 107
    for(;;)
    {

#line 107
        if(i_2 < num_features_2)
        {
        }
        else
        {

#line 107
            break;
        }

#line 108
        float _S36 = ((output_grad_0).load<float>((_S29), (uint(i_2)), (_S30), (_S31)));

#line 108
        float _S37 = - _S36;

#line 108
        float _S38 = ((output_2).load<float>((_S29), (uint(i_2)), (_S30), (_S31)));

#line 108
        float dLdTotalWeight_1 = dLdTotalWeight_0 + _S37 * _S38 / _S32;

#line 107
        i_2 = i_2 + int(1);

#line 107
        dLdTotalWeight_0 = dLdTotalWeight_1;

#line 107
    }

#line 107
    int yoff_1 = _S33;



    for(;;)
    {

#line 111
        if(yoff_1 <= kernel_boundary_2)
        {
        }
        else
        {

#line 111
            break;
        }

#line 111
        int xoff_1 = _S33;
        for(;;)
        {

#line 112
            if(xoff_1 <= kernel_boundary_2)
            {
            }
            else
            {

#line 112
                break;
            }

#line 113
            int xp_1 = x_2 + xoff_1 * dialation_2;
            int yp_1 = y_2 + yoff_1 * dialation_2;

#line 114
            bool _S39;


            if(xp_1 < int(0))
            {

#line 117
                _S39 = true;

#line 117
            }
            else
            {

#line 117
                _S39 = yp_1 < int(0);

#line 117
            }

#line 117
            bool _S40;

#line 117
            if(_S39)
            {

#line 117
                _S40 = true;

#line 117
            }
            else
            {

#line 117
                _S40 = xp_1 >= width_2;

#line 117
            }

#line 117
            bool _S41;

#line 117
            if(_S40)
            {

#line 117
                _S41 = true;

#line 117
            }
            else
            {

#line 117
                _S41 = yp_1 >= height_2;

#line 117
            }

#line 117
            if(_S41)
            {

#line 118
                xoff_1 = xoff_1 + int(1);

#line 112
                continue;
            }

#line 112
            float center_val_1;

#line 112
            float sample_val_1;

#line 112
            int i_3;

#line 112
            i_2 = int(0);

#line 112
            float sample_weight_2 = 1.0f;

#line 127
            uint _S42 = uint(yp_1);

#line 127
            uint _S43 = uint(xp_1);

#line 122
            for(;;)
            {

#line 122
                if(i_2 < _S34)
                {
                }
                else
                {

#line 122
                    break;
                }

                if(i_2 < num_features_2)
                {

#line 126
                    float _S44 = ((input_2).load<float>((_S29), (uint(i_2)), (_S30), (_S31)));
                    float _S45 = ((input_2).load<float>((_S29), (uint(i_2)), (_S42), (_S43)));

#line 127
                    sample_val_1 = _S45;

#line 127
                    center_val_1 = _S44;

#line 125
                }
                else
                {


                    if(i_2 == num_features_2)
                    {

#line 130
                        i_3 = yoff_1;

#line 130
                    }
                    else
                    {

#line 130
                        i_3 = xoff_1;

#line 130
                    }

#line 130
                    sample_val_1 = float(i_3);

#line 130
                    center_val_1 = 0.0f;

#line 125
                }

#line 133
                float diff_1 = sample_val_1 - center_val_1;

                float _S46 = ((params_2).load<float>((uint(i_2))));

                float sample_weight_3 = sample_weight_2 * (F32_exp((_S46 * diff_1 * diff_1)));

#line 122
                i_2 = i_2 + int(1);

#line 122
                sample_weight_2 = sample_weight_3;

#line 122
            }

#line 122
            i_3 = int(0);

#line 122
            float dLdSampleWeight_0 = dLdTotalWeight_0;

#line 141
            for(;;)
            {

#line 141
                if(i_3 < num_features_2)
                {
                }
                else
                {

#line 141
                    break;
                }

#line 142
                float _S47 = ((output_grad_0).load<float>((_S29), (uint(i_3)), (_S30), (_S31)));

#line 142
                float _S48 = ((input_2).load<float>((_S29), (uint(i_3)), (_S42), (_S43)));

#line 142
                float dLdSampleWeight_1 = dLdSampleWeight_0 + _S47 * _S48 / _S28;

#line 141
                i_3 = i_3 + int(1);

#line 141
                dLdSampleWeight_0 = dLdSampleWeight_1;

#line 141
            }

#line 141
            int i_4 = int(0);

#line 146
            for(;;)
            {

#line 146
                if(i_4 < _S34)
                {
                }
                else
                {

#line 146
                    break;
                }

                if(i_4 < num_features_2)
                {

#line 150
                    float _S49 = ((input_2).load<float>((_S29), (uint(i_4)), (_S30), (_S31)));
                    float _S50 = ((input_2).load<float>((_S29), (uint(i_4)), (_S42), (_S43)));

#line 151
                    sample_val_1 = _S50;

#line 151
                    center_val_1 = _S49;

#line 149
                }
                else
                {

#line 149
                    int _S51;

#line 154
                    if(i_4 == num_features_2)
                    {

#line 154
                        _S51 = yoff_1;

#line 154
                    }
                    else
                    {

#line 154
                        _S51 = xoff_1;

#line 154
                    }

#line 154
                    sample_val_1 = float(_S51);

#line 154
                    center_val_1 = 0.0f;

#line 149
                }

#line 157
                float diff_2 = sample_val_1 - center_val_1;


                float _S52 = atomicAdd(&(*&shr_dLdParam_0)[_S26][i_4], dLdSampleWeight_0 * sample_weight_2 * diff_2 * diff_2);


                if(i_4 < num_features_2)
                {

                    float _S53 = dLdSampleWeight_0 * sample_weight_2;

#line 166
                    float _S54 = ((params_2).load<float>((uint(i_4))));

#line 166
                    float dLdCenterVal_0 = _S53 * _S54 * -2.0f * diff_2;
                    int4  _S55 = make_int4 (b_2, i_4, y_2, x_2);

#line 167
                    uint4  _S56 = make_uint4 ((uint)_S55.x, (uint)_S55.y, (uint)_S55.z, (uint)_S55.w);

#line 164
                    float junk_0;


                    *((&junk_0)) = atomicAdd((input_grad_0).data_ptr_at<float>((_S56)), (dLdCenterVal_0));

                    float _S57 = ((output_grad_0).load<float>((_S29), (uint(i_4)), (_S30), (_S31)));
                    int4  _S58 = make_int4 (b_2, i_4, yp_1, xp_1);

#line 170
                    uint4  _S59 = make_uint4 ((uint)_S58.x, (uint)_S58.y, (uint)_S58.z, (uint)_S58.w);

#line 170
                    *((&junk_0)) = atomicAdd((input_grad_0).data_ptr_at<float>((_S59)), (_S57 * sample_weight_2 / _S28 - dLdCenterVal_0));

#line 163
                }

#line 146
                i_4 = i_4 + int(1);

#line 146
            }

#line 112
            xoff_1 = xoff_1 + int(1);

#line 112
        }

#line 111
        yoff_1 = yoff_1 + int(1);

#line 111
    }

#line 177
    __syncthreads();

    if(_S35)
    {

#line 179
        i_2 = int(1);

#line 198
        uint _S60 = uint(j_0);

#line 180
        for(;;)
        {

#line 180
            if(i_2 < int(32))
            {
            }
            else
            {

#line 180
                break;
            }
            (*&shr_dLdParam_0)[int(0)][j_0] = (*&shr_dLdParam_0)[int(0)][j_0] + (*&shr_dLdParam_0)[i_2][j_0];

#line 180
            i_2 = i_2 + int(1);

#line 180
        }

#line 197
        float junk_1;
        *((&junk_1)) = atomicAdd((params_grad_0).data_ptr_at<float>((_S60)), ((*&shr_dLdParam_0)[int(0)][j_0]));

#line 179
    }

#line 201
    return;
}


#line 633
__global__ void __kernel__bwd_bilateral_filter_wrapper(TensorView input_3, TensorView input_grad_1, TensorView params_3, TensorView params_grad_1, TensorView output_3, TensorView output_grad_1, int kernel_boundary_3, int dialation_3)
{

#line 605
    int globalIdx_1 = int(((threadIdx)).x + ((blockIdx)).x * ((blockDim)).x);
    uint _S61 = ((input_3).sizes[(0U)]);

#line 606
    int batch_size_1 = int(_S61);
    uint _S62 = ((input_3).sizes[(1U)]);

#line 607
    int num_features_3 = int(_S62);
    uint _S63 = ((input_3).sizes[(3U)]);

#line 608
    int width_3 = int(_S63);
    uint _S64 = ((input_3).sizes[(2U)]);

#line 609
    int height_3 = int(_S64);
    if(globalIdx_1 >= batch_size_1 * width_3 * height_3)
    {

#line 611
        return;
    }
    int x_3 = globalIdx_1 % width_3;
    int _S65 = globalIdx_1 / width_3;

#line 614
    int y_3 = _S65 % height_3;
    int b_3 = globalIdx_1 / (width_3 * height_3);

#line 645
    bwd_bilteral_filter_0(num_features_3, width_3, height_3, b_3, y_3, x_3, input_3, input_grad_1, params_3, params_grad_1, output_3, output_grad_1, kernel_boundary_3, dialation_3);
    return;
}


#line 204
__device__ float kernel_bilteral_filter_0(int num_features_4, int width_4, int height_4, int b_4, int y_4, int x_4, TensorView input_4, TensorView params_4, TensorView kernel_0, TensorView output_4, int kernel_boundary_4, int dialation_4, bool normalize_1)
{

#line 217
    int xoff_2;



    int _S66 = - kernel_boundary_4;

#line 221
    int yoff_2 = _S66;

#line 221
    float total_weight_2 = 0.0f;

#line 232
    int _S67 = num_features_4 + int(2);



    uint _S68 = uint(b_4);

#line 236
    uint _S69 = uint(y_4);

#line 236
    uint _S70 = uint(x_4);

#line 221
    for(;;)
    {

#line 221
        if(yoff_2 <= kernel_boundary_4)
        {
        }
        else
        {

#line 221
            break;
        }

#line 221
        xoff_2 = _S66;

#line 221
        float total_weight_3 = total_weight_2;
        for(;;)
        {

#line 222
            if(xoff_2 <= kernel_boundary_4)
            {
            }
            else
            {

#line 222
                break;
            }

#line 223
            int xp_2 = x_4 + xoff_2 * dialation_4;
            int yp_2 = y_4 + yoff_2 * dialation_4;

#line 224
            bool _S71;


            if(xp_2 < int(0))
            {

#line 227
                _S71 = true;

#line 227
            }
            else
            {

#line 227
                _S71 = yp_2 < int(0);

#line 227
            }

#line 227
            bool _S72;

#line 227
            if(_S71)
            {

#line 227
                _S72 = true;

#line 227
            }
            else
            {

#line 227
                _S72 = xp_2 >= width_4;

#line 227
            }

#line 227
            bool _S73;

#line 227
            if(_S72)
            {

#line 227
                _S73 = true;

#line 227
            }
            else
            {

#line 227
                _S73 = yp_2 >= height_4;

#line 227
            }

#line 227
            if(_S73)
            {

#line 228
                xoff_2 = xoff_2 + int(1);

#line 222
                continue;
            }

#line 222
            int i_5;

#line 231
            float _S74 = ((kernel_0).load<float>((uint(yoff_2 + kernel_boundary_4)), (uint(xoff_2 + kernel_boundary_4))));

#line 231
            int i_6 = int(0);

#line 231
            float sample_weight_4 = _S74;

#line 237
            uint _S75 = uint(yp_2);

#line 237
            uint _S76 = uint(xp_2);

#line 232
            for(;;)
            {

#line 232
                if(i_6 < _S67)
                {
                }
                else
                {

#line 232
                    break;
                }

#line 232
                float sample_val_2;

#line 232
                float center_val_2;


                if(i_6 < num_features_4)
                {

#line 236
                    float _S77 = ((input_4).load<float>((_S68), (uint(i_6)), (_S69), (_S70)));
                    float _S78 = ((input_4).load<float>((_S68), (uint(i_6)), (_S75), (_S76)));

#line 237
                    sample_val_2 = _S78;

#line 237
                    center_val_2 = _S77;

#line 235
                }
                else
                {


                    if(i_6 == num_features_4)
                    {

#line 240
                        i_5 = yoff_2;

#line 240
                    }
                    else
                    {

#line 240
                        i_5 = xoff_2;

#line 240
                    }

#line 240
                    sample_val_2 = float(i_5);

#line 240
                    center_val_2 = 0.0f;

#line 235
                }

#line 243
                float diff_3 = sample_val_2 - center_val_2;

#line 251
                float _S79 = ((params_4).load<float>((uint(i_6))));

                float sample_weight_5 = sample_weight_4 * (F32_exp((_S79 * diff_3 * diff_3)));

#line 232
                i_6 = i_6 + int(1);

#line 232
                sample_weight_4 = sample_weight_5;

#line 232
            }

#line 232
            i_5 = int(0);

#line 256
            for(;;)
            {

#line 256
                if(i_5 < num_features_4)
                {
                }
                else
                {

#line 256
                    break;
                }

#line 257
                float _S80 = ((input_4).load<float>((_S68), (uint(i_5)), (_S75), (_S76)));
                uint _S81 = uint(i_5);

#line 258
                float _S82 = ((output_4).load<float>((_S68), (_S81), (_S69), (_S70)));

#line 258
                (output_4).store<float>((_S68), (_S81), (_S69), (_S70), (_S82 + sample_weight_4 * _S80));

#line 256
                i_5 = i_5 + int(1);

#line 256
            }

#line 256
            total_weight_3 = total_weight_3 + sample_weight_4;

#line 222
            xoff_2 = xoff_2 + int(1);

#line 222
        }

#line 221
        yoff_2 = yoff_2 + int(1);

#line 221
        total_weight_2 = total_weight_3;

#line 221
    }

#line 265
    if(total_weight_2 < 0.00100000004749745f)
    {

#line 265
        total_weight_2 = 0.00100000004749745f;

#line 265
    }

    if(normalize_1)
    {

#line 267
        xoff_2 = int(0);
        for(;;)
        {

#line 268
            if(xoff_2 < num_features_4)
            {
            }
            else
            {

#line 268
                break;
            }

#line 269
            uint _S83 = uint(xoff_2);

#line 269
            float _S84 = ((output_4).load<float>((_S68), (_S83), (_S69), (_S70)));

#line 269
            (output_4).store<float>((_S68), (_S83), (_S69), (_S70), (_S84 / total_weight_2));

#line 268
            xoff_2 = xoff_2 + int(1);

#line 268
        }

#line 267
    }

#line 273
    return total_weight_2;
}


#line 650
__global__ void __kernel__exec_kernel_bilateral_filter_wrapper(TensorView input_5, TensorView params_5, TensorView kernel_1, TensorView output_5, int kernel_boundary_5, int dialation_5)
{

#line 605
    int globalIdx_2 = int(((threadIdx)).x + ((blockIdx)).x * ((blockDim)).x);
    uint _S85 = ((input_5).sizes[(0U)]);

#line 606
    int batch_size_2 = int(_S85);
    uint _S86 = ((input_5).sizes[(1U)]);

#line 607
    int num_features_5 = int(_S86);
    uint _S87 = ((input_5).sizes[(3U)]);

#line 608
    int width_5 = int(_S87);
    uint _S88 = ((input_5).sizes[(2U)]);

#line 609
    int height_5 = int(_S88);
    if(globalIdx_2 >= batch_size_2 * width_5 * height_5)
    {

#line 611
        return;
    }
    int x_5 = globalIdx_2 % width_5;
    int _S89 = globalIdx_2 / width_5;

#line 614
    int y_5 = _S89 % height_5;
    int b_5 = globalIdx_2 / (width_5 * height_5);

#line 660
    float _S90 = kernel_bilteral_filter_0(num_features_5, width_5, height_5, b_5, y_5, x_5, input_5, params_5, kernel_1, output_5, kernel_boundary_5, dialation_5, true);
    return;
}


#line 278
__device__ __shared__ FixedArray<FixedArray<FixedArray<float, 7> , 7> , 16>  shr_dLdKernel_0;

__device__ void bwd_kernel_bilteral_filter_0(int num_features_6, int width_6, int height_6, int b_6, int y_6, int x_6, TensorView input_6, TensorView input_grad_2, TensorView params_6, TensorView params_grad_2, TensorView kernel_2, TensorView kernel_grad_0, TensorView output_6, TensorView output_grad_2, int kernel_boundary_6, int dialation_6)
{

#line 298
    uint _S91 = ((threadIdx)).x;

#line 298
    int _S92 = int(_S91 % 32U);
    int _S93 = int(_S91 % 16U);

#line 299
    int i_7;

    if(_S91 < 32U)
    {

#line 301
        i_7 = int(0);
        int _S94 = num_features_6 + int(2);

#line 302
        for(;;)
        {

#line 302
            if(i_7 < _S94)
            {
            }
            else
            {

#line 302
                break;
            }

#line 303
            (*&shr_dLdParam_0)[_S91][i_7] = 0.0f;

#line 302
            i_7 = i_7 + int(1);

#line 302
        }

#line 301
    }

#line 301
    int yoff_3;

#line 307
    if(_S91 < 16U)
    {

#line 307
        i_7 = int(0);
        for(;;)
        {

#line 308
            if(i_7 < int(7))
            {
            }
            else
            {

#line 308
                break;
            }

#line 308
            yoff_3 = int(0);
            for(;;)
            {

#line 309
                if(yoff_3 < int(7))
                {
                }
                else
                {

#line 309
                    break;
                }

#line 310
                (*&shr_dLdKernel_0)[_S91][i_7][yoff_3] = 0.0f;

#line 309
                yoff_3 = yoff_3 + int(1);

#line 309
            }

#line 308
            i_7 = i_7 + int(1);

#line 308
        }

#line 307
    }

#line 316
    float _S95 = kernel_bilteral_filter_0(num_features_6, width_6, height_6, b_6, y_6, x_6, input_6, params_6, kernel_2, output_6, kernel_boundary_6, dialation_6, false);

#line 316
    i_7 = int(0);

#line 316
    float dLdTotalWeight_2 = 0.0f;



    uint _S96 = uint(b_6);

#line 320
    uint _S97 = uint(y_6);

#line 320
    uint _S98 = uint(x_6);

#line 320
    float _S99 = _S95 * _S95;


    int _S100 = - kernel_boundary_6;

#line 394
    int j_1 = int(_S91);
    int _S101 = num_features_6 + int(2);

#line 395
    bool _S102 = j_1 < _S101;

#line 319
    for(;;)
    {

#line 319
        if(i_7 < num_features_6)
        {
        }
        else
        {

#line 319
            break;
        }

#line 320
        float _S103 = ((output_grad_2).load<float>((_S96), (uint(i_7)), (_S97), (_S98)));

#line 320
        float _S104 = - _S103;

#line 320
        float _S105 = ((output_6).load<float>((_S96), (uint(i_7)), (_S97), (_S98)));

#line 320
        float dLdTotalWeight_3 = dLdTotalWeight_2 + _S104 * _S105 / _S99;

#line 319
        i_7 = i_7 + int(1);

#line 319
        dLdTotalWeight_2 = dLdTotalWeight_3;

#line 319
    }

#line 319
    float sample_weight_6;

#line 319
    yoff_3 = _S100;



    for(;;)
    {

#line 323
        if(yoff_3 <= kernel_boundary_6)
        {
        }
        else
        {

#line 323
            break;
        }

#line 323
        int xoff_3 = _S100;
        for(;;)
        {

#line 324
            if(xoff_3 <= kernel_boundary_6)
            {
            }
            else
            {

#line 324
                break;
            }

#line 325
            int xp_3 = x_6 + xoff_3 * dialation_6;
            int yp_3 = y_6 + yoff_3 * dialation_6;

#line 326
            bool _S106;


            if(xp_3 < int(0))
            {

#line 329
                _S106 = true;

#line 329
            }
            else
            {

#line 329
                _S106 = yp_3 < int(0);

#line 329
            }

#line 329
            bool _S107;

#line 329
            if(_S106)
            {

#line 329
                _S107 = true;

#line 329
            }
            else
            {

#line 329
                _S107 = xp_3 >= width_6;

#line 329
            }

#line 329
            bool _S108;

#line 329
            if(_S107)
            {

#line 329
                _S108 = true;

#line 329
            }
            else
            {

#line 329
                _S108 = yp_3 >= height_6;

#line 329
            }

#line 329
            if(_S108)
            {

#line 330
                xoff_3 = xoff_3 + int(1);

#line 324
                continue;
            }

#line 324
            float center_val_3;

#line 324
            float sample_val_3;

#line 324
            int i_8;

#line 333
            float _S109 = ((kernel_2).load<float>((uint(yoff_3 + kernel_boundary_6)), (uint(xoff_3 + kernel_boundary_6))));

#line 333
            i_7 = int(0);

#line 333
            sample_weight_6 = _S109;

#line 333
            float non_kernel_weight_0 = 1.0f;

#line 340
            uint _S110 = uint(yp_3);

#line 340
            uint _S111 = uint(xp_3);

#line 335
            for(;;)
            {

#line 335
                if(i_7 < _S101)
                {
                }
                else
                {

#line 335
                    break;
                }

                if(i_7 < num_features_6)
                {

#line 339
                    float _S112 = ((input_6).load<float>((_S96), (uint(i_7)), (_S97), (_S98)));
                    float _S113 = ((input_6).load<float>((_S96), (uint(i_7)), (_S110), (_S111)));

#line 340
                    sample_val_3 = _S113;

#line 340
                    center_val_3 = _S112;

#line 338
                }
                else
                {


                    if(i_7 == num_features_6)
                    {

#line 343
                        i_8 = yoff_3;

#line 343
                    }
                    else
                    {

#line 343
                        i_8 = xoff_3;

#line 343
                    }

#line 343
                    sample_val_3 = float(i_8);

#line 343
                    center_val_3 = 0.0f;

#line 338
                }

#line 346
                float diff_4 = sample_val_3 - center_val_3;

                float _S114 = ((params_6).load<float>((uint(i_7))));

#line 348
                float weight_0 = (F32_exp((_S114 * diff_4 * diff_4)));

                float sample_weight_7 = sample_weight_6 * weight_0;
                float non_kernel_weight_1 = non_kernel_weight_0 * weight_0;

#line 335
                i_7 = i_7 + int(1);

#line 335
                sample_weight_6 = sample_weight_7;

#line 335
                non_kernel_weight_0 = non_kernel_weight_1;

#line 335
            }

#line 335
            i_8 = int(0);

#line 335
            float dLdSampleWeight_2 = dLdTotalWeight_2;

#line 355
            for(;;)
            {

#line 355
                if(i_8 < num_features_6)
                {
                }
                else
                {

#line 355
                    break;
                }

#line 356
                float _S115 = ((output_grad_2).load<float>((_S96), (uint(i_8)), (_S97), (_S98)));

#line 356
                float _S116 = ((input_6).load<float>((_S96), (uint(i_8)), (_S110), (_S111)));

#line 356
                float dLdSampleWeight_3 = dLdSampleWeight_2 + _S115 * _S116 / _S95;

#line 355
                i_8 = i_8 + int(1);

#line 355
                dLdSampleWeight_2 = dLdSampleWeight_3;

#line 355
            }

#line 360
            float _S117 = atomicAdd(&(*&shr_dLdKernel_0)[_S93][yoff_3 + kernel_boundary_6][xoff_3 + kernel_boundary_6], non_kernel_weight_0 * dLdSampleWeight_2);

#line 360
            int i_9 = int(0);


            for(;;)
            {

#line 363
                if(i_9 < _S101)
                {
                }
                else
                {

#line 363
                    break;
                }

                if(i_9 < num_features_6)
                {

#line 367
                    float _S118 = ((input_6).load<float>((_S96), (uint(i_9)), (_S97), (_S98)));
                    float _S119 = ((input_6).load<float>((_S96), (uint(i_9)), (_S110), (_S111)));

#line 368
                    sample_val_3 = _S119;

#line 368
                    center_val_3 = _S118;

#line 366
                }
                else
                {

#line 366
                    int _S120;

#line 371
                    if(i_9 == num_features_6)
                    {

#line 371
                        _S120 = yoff_3;

#line 371
                    }
                    else
                    {

#line 371
                        _S120 = xoff_3;

#line 371
                    }

#line 371
                    sample_val_3 = float(_S120);

#line 371
                    center_val_3 = 0.0f;

#line 366
                }

#line 374
                float diff_5 = sample_val_3 - center_val_3;


                float _S121 = atomicAdd(&(*&shr_dLdParam_0)[_S92][i_9], dLdSampleWeight_2 * sample_weight_6 * diff_5 * diff_5);


                if(i_9 < num_features_6)
                {

                    float _S122 = dLdSampleWeight_2 * sample_weight_6;

#line 383
                    float _S123 = ((params_6).load<float>((uint(i_9))));

#line 383
                    float dLdCenterVal_1 = _S122 * _S123 * -2.0f * diff_5;
                    int4  _S124 = make_int4 (b_6, i_9, y_6, x_6);

#line 384
                    uint4  _S125 = make_uint4 ((uint)_S124.x, (uint)_S124.y, (uint)_S124.z, (uint)_S124.w);

#line 381
                    float junk_2;


                    *((&junk_2)) = atomicAdd((input_grad_2).data_ptr_at<float>((_S125)), (dLdCenterVal_1));

                    float _S126 = ((output_grad_2).load<float>((_S96), (uint(i_9)), (_S97), (_S98)));
                    int4  _S127 = make_int4 (b_6, i_9, yp_3, xp_3);

#line 387
                    uint4  _S128 = make_uint4 ((uint)_S127.x, (uint)_S127.y, (uint)_S127.z, (uint)_S127.w);

#line 387
                    *((&junk_2)) = atomicAdd((input_grad_2).data_ptr_at<float>((_S128)), (_S126 * sample_weight_6 / _S95 - dLdCenterVal_1));

#line 380
                }

#line 363
                i_9 = i_9 + int(1);

#line 363
            }

#line 324
            xoff_3 = xoff_3 + int(1);

#line 324
        }

#line 323
        yoff_3 = yoff_3 + int(1);

#line 323
    }

#line 393
    __syncthreads();

    if(_S102)
    {

#line 395
        i_7 = int(1);

#line 414
        uint _S129 = uint(j_1);

#line 396
        for(;;)
        {

#line 396
            if(i_7 < int(32))
            {
            }
            else
            {

#line 396
                break;
            }
            (*&shr_dLdParam_0)[int(0)][j_1] = (*&shr_dLdParam_0)[int(0)][j_1] + (*&shr_dLdParam_0)[i_7][j_1];

#line 396
            i_7 = i_7 + int(1);

#line 396
        }

#line 413
        float junk_3;
        *((&junk_3)) = atomicAdd((params_grad_2).data_ptr_at<float>((_S129)), ((*&shr_dLdParam_0)[int(0)][j_1]));

#line 395
    }

#line 417
    uint _S130 = uint(j_1);

#line 417
    uint _S131 = ((kernel_2).sizes[(0U)]);

#line 417
    uint _S132 = ((kernel_2).sizes[(0U)]);

#line 417
    if(_S130 < _S131 * _S132)
    {

#line 418
        uint _S133 = ((kernel_2).sizes[(0U)]);

#line 418
        uint _S134 = _S130 / _S133;

#line 418
        int y_7 = int(_S134);
        uint _S135 = ((kernel_2).sizes[(0U)]);

#line 419
        uint _S136 = _S130 % _S135;

#line 419
        int x_7 = int(_S136);

#line 419
        i_7 = int(0);

#line 419
        sample_weight_6 = 0.0f;

#line 427
        uint2  _S137 = make_uint2 (uint(y_7), uint(x_7));

#line 422
        for(;;)
        {

#line 422
            if(i_7 < int(16))
            {
            }
            else
            {

#line 422
                break;
            }

#line 423
            float accum_0 = sample_weight_6 + (*&shr_dLdKernel_0)[i_7][y_7][x_7];

#line 422
            i_7 = i_7 + int(1);

#line 422
            sample_weight_6 = accum_0;

#line 422
        }



        float junk_4;
        *((&junk_4)) = atomicAdd((kernel_grad_0).data_ptr_at<float>((_S137)), (sample_weight_6));

#line 417
    }

#line 429
    return;
}


#line 665
__global__ void __kernel__bwd_kernel_bilateral_filter_wrapper(TensorView input_7, TensorView input_grad_3, TensorView params_7, TensorView params_grad_3, TensorView kernel_3, TensorView kernel_grad_1, TensorView output_7, TensorView output_grad_3, int kernel_boundary_7, int dialation_7)
{

#line 605
    int globalIdx_3 = int(((threadIdx)).x + ((blockIdx)).x * ((blockDim)).x);
    uint _S138 = ((input_7).sizes[(0U)]);

#line 606
    int batch_size_3 = int(_S138);
    uint _S139 = ((input_7).sizes[(1U)]);

#line 607
    int num_features_7 = int(_S139);
    uint _S140 = ((input_7).sizes[(3U)]);

#line 608
    int width_7 = int(_S140);
    uint _S141 = ((input_7).sizes[(2U)]);

#line 609
    int height_7 = int(_S141);
    if(globalIdx_3 >= batch_size_3 * width_7 * height_7)
    {

#line 611
        return;
    }
    int x_8 = globalIdx_3 % width_7;
    int _S142 = globalIdx_3 / width_7;

#line 614
    int y_8 = _S142 % height_7;
    int b_7 = globalIdx_3 / (width_7 * height_7);

#line 679
    bwd_kernel_bilteral_filter_0(num_features_7, width_7, height_7, b_7, y_8, x_8, input_7, input_grad_3, params_7, params_grad_3, kernel_3, kernel_grad_1, output_7, output_grad_3, kernel_boundary_7, dialation_7);
    return;
}


#line 438
__device__ float pixel_bilteral_filter_0(int num_features_8, int width_8, int height_8, int b_8, int y_9, int x_9, TensorView input_8, TensorView params_8, TensorView output_8, int kernel_boundary_8, int dialation_8, bool normalize_2)
{

#line 450
    int xoff_4;



    int _S143 = - kernel_boundary_8;

#line 454
    int yoff_4 = _S143;

#line 454
    float total_weight_4 = 0.0f;

#line 465
    int _S144 = num_features_8 + int(2);

#line 484
    uint _S145 = uint(b_8);

#line 484
    uint _S146 = uint(y_9);

#line 484
    uint _S147 = uint(x_9);

#line 454
    for(;;)
    {

#line 454
        if(yoff_4 <= kernel_boundary_8)
        {
        }
        else
        {

#line 454
            break;
        }

#line 454
        xoff_4 = _S143;

#line 454
        float total_weight_5 = total_weight_4;
        for(;;)
        {

#line 455
            if(xoff_4 <= kernel_boundary_8)
            {
            }
            else
            {

#line 455
                break;
            }

#line 456
            int xp_4 = x_9 + xoff_4 * dialation_8;
            int yp_4 = y_9 + yoff_4 * dialation_8;

#line 457
            bool _S148;


            if(xp_4 < int(0))
            {

#line 460
                _S148 = true;

#line 460
            }
            else
            {

#line 460
                _S148 = yp_4 < int(0);

#line 460
            }

#line 460
            bool _S149;

#line 460
            if(_S148)
            {

#line 460
                _S149 = true;

#line 460
            }
            else
            {

#line 460
                _S149 = xp_4 >= width_8;

#line 460
            }

#line 460
            bool _S150;

#line 460
            if(_S149)
            {

#line 460
                _S150 = true;

#line 460
            }
            else
            {

#line 460
                _S150 = yp_4 >= height_8;

#line 460
            }

#line 460
            if(_S150)
            {

#line 461
                xoff_4 = xoff_4 + int(1);

#line 455
                continue;
            }

#line 455
            int i_10;

#line 455
            int i_11 = int(0);

#line 455
            float sample_weight_8 = 1.0f;

#line 470
            uint _S151 = uint(yp_4);

#line 470
            uint _S152 = uint(xp_4);

#line 465
            for(;;)
            {

#line 465
                if(i_11 < _S144)
                {
                }
                else
                {

#line 465
                    break;
                }

#line 465
                float sample_val_4;

#line 465
                float center_val_4;


                if(i_11 < num_features_8)
                {

#line 469
                    float _S153 = ((input_8).load<float>((_S145), (uint(i_11)), (_S146), (_S147)));
                    float _S154 = ((input_8).load<float>((_S145), (uint(i_11)), (_S151), (_S152)));

#line 470
                    sample_val_4 = _S154;

#line 470
                    center_val_4 = _S153;

#line 468
                }
                else
                {


                    if(i_11 == num_features_8)
                    {

#line 473
                        i_10 = yoff_4;

#line 473
                    }
                    else
                    {

#line 473
                        i_10 = xoff_4;

#line 473
                    }

#line 473
                    sample_val_4 = float(i_10);

#line 473
                    center_val_4 = 0.0f;

#line 468
                }

#line 476
                float diff_6 = sample_val_4 - center_val_4;

#line 484
                float _S155 = ((params_8).load<float>((_S145), (uint(i_11)), (_S146), (_S147)));

                float sample_weight_9 = sample_weight_8 * (F32_exp((_S155 * diff_6 * diff_6)));

#line 465
                i_11 = i_11 + int(1);

#line 465
                sample_weight_8 = sample_weight_9;

#line 465
            }

#line 465
            i_10 = int(0);

#line 489
            for(;;)
            {

#line 489
                if(i_10 < num_features_8)
                {
                }
                else
                {

#line 489
                    break;
                }

#line 490
                float _S156 = ((input_8).load<float>((_S145), (uint(i_10)), (_S151), (_S152)));
                uint _S157 = uint(i_10);

#line 491
                float _S158 = ((output_8).load<float>((_S145), (_S157), (_S146), (_S147)));

#line 491
                (output_8).store<float>((_S145), (_S157), (_S146), (_S147), (_S158 + sample_weight_8 * _S156));

#line 489
                i_10 = i_10 + int(1);

#line 489
            }

#line 489
            total_weight_5 = total_weight_5 + sample_weight_8;

#line 455
            xoff_4 = xoff_4 + int(1);

#line 455
        }

#line 454
        yoff_4 = yoff_4 + int(1);

#line 454
        total_weight_4 = total_weight_5;

#line 454
    }

#line 498
    if(total_weight_4 < 0.00100000004749745f)
    {

#line 498
        total_weight_4 = 0.00100000004749745f;

#line 498
    }

    if(normalize_2)
    {

#line 500
        xoff_4 = int(0);
        for(;;)
        {

#line 501
            if(xoff_4 < num_features_8)
            {
            }
            else
            {

#line 501
                break;
            }

#line 502
            uint _S159 = uint(xoff_4);

#line 502
            float _S160 = ((output_8).load<float>((_S145), (_S159), (_S146), (_S147)));

#line 502
            (output_8).store<float>((_S145), (_S159), (_S146), (_S147), (_S160 / total_weight_4));

#line 501
            xoff_4 = xoff_4 + int(1);

#line 501
        }

#line 500
    }

#line 506
    return total_weight_4;
}


#line 684
__global__ void __kernel__exec_pixel_bilateral_filter_wrapper(TensorView input_9, TensorView params_9, TensorView output_9, int kernel_boundary_9, int dialation_9)
{

#line 605
    int globalIdx_4 = int(((threadIdx)).x + ((blockIdx)).x * ((blockDim)).x);
    uint _S161 = ((input_9).sizes[(0U)]);

#line 606
    int batch_size_4 = int(_S161);
    uint _S162 = ((input_9).sizes[(1U)]);

#line 607
    int num_features_9 = int(_S162);
    uint _S163 = ((input_9).sizes[(3U)]);

#line 608
    int width_9 = int(_S163);
    uint _S164 = ((input_9).sizes[(2U)]);

#line 609
    int height_9 = int(_S164);
    if(globalIdx_4 >= batch_size_4 * width_9 * height_9)
    {

#line 611
        return;
    }
    int x_10 = globalIdx_4 % width_9;
    int _S165 = globalIdx_4 / width_9;

#line 614
    int y_10 = _S165 % height_9;
    int b_9 = globalIdx_4 / (width_9 * height_9);

#line 693
    float _S166 = pixel_bilteral_filter_0(num_features_9, width_9, height_9, b_9, y_10, x_10, input_9, params_9, output_9, kernel_boundary_9, dialation_9, true);
    return;
}


#line 509
__device__ void bwd_pixel_bilteral_filter_0(int num_features_10, int width_10, int height_10, int b_10, int y_11, int x_11, TensorView input_10, TensorView input_grad_4, TensorView params_10, TensorView params_grad_4, TensorView output_10, TensorView output_grad_4, int kernel_boundary_10, int dialation_10)
{

#line 526
    float _S167 = pixel_bilteral_filter_0(num_features_10, width_10, height_10, b_10, y_11, x_11, input_10, params_10, output_10, kernel_boundary_10, dialation_10, false);

#line 526
    int i_12 = int(0);

#line 526
    float dLdTotalWeight_4 = 0.0f;



    uint _S168 = uint(b_10);

#line 530
    uint _S169 = uint(y_11);

#line 530
    uint _S170 = uint(x_11);

#line 530
    float _S171 = _S167 * _S167;


    int _S172 = - kernel_boundary_10;

#line 544
    int _S173 = num_features_10 + int(2);

#line 529
    for(;;)
    {

#line 529
        if(i_12 < num_features_10)
        {
        }
        else
        {

#line 529
            break;
        }

#line 530
        float _S174 = ((output_grad_4).load<float>((_S168), (uint(i_12)), (_S169), (_S170)));

#line 530
        float _S175 = - _S174;

#line 530
        float _S176 = ((output_10).load<float>((_S168), (uint(i_12)), (_S169), (_S170)));

#line 530
        float dLdTotalWeight_5 = dLdTotalWeight_4 + _S175 * _S176 / _S171;

#line 529
        i_12 = i_12 + int(1);

#line 529
        dLdTotalWeight_4 = dLdTotalWeight_5;

#line 529
    }

#line 529
    int yoff_5 = _S172;



    for(;;)
    {

#line 533
        if(yoff_5 <= kernel_boundary_10)
        {
        }
        else
        {

#line 533
            break;
        }

#line 533
        int xoff_5 = _S172;
        for(;;)
        {

#line 534
            if(xoff_5 <= kernel_boundary_10)
            {
            }
            else
            {

#line 534
                break;
            }

#line 535
            int xp_5 = x_11 + xoff_5 * dialation_10;
            int yp_5 = y_11 + yoff_5 * dialation_10;

#line 536
            bool _S177;


            if(xp_5 < int(0))
            {

#line 539
                _S177 = true;

#line 539
            }
            else
            {

#line 539
                _S177 = yp_5 < int(0);

#line 539
            }

#line 539
            bool _S178;

#line 539
            if(_S177)
            {

#line 539
                _S178 = true;

#line 539
            }
            else
            {

#line 539
                _S178 = xp_5 >= width_10;

#line 539
            }

#line 539
            bool _S179;

#line 539
            if(_S178)
            {

#line 539
                _S179 = true;

#line 539
            }
            else
            {

#line 539
                _S179 = yp_5 >= height_10;

#line 539
            }

#line 539
            if(_S179)
            {

#line 540
                xoff_5 = xoff_5 + int(1);

#line 534
                continue;
            }

#line 534
            float center_val_5;

#line 534
            float sample_val_5;

#line 534
            int i_13;

#line 534
            i_12 = int(0);

#line 534
            float sample_weight_10 = 1.0f;

#line 549
            uint _S180 = uint(yp_5);

#line 549
            uint _S181 = uint(xp_5);

#line 544
            for(;;)
            {

#line 544
                if(i_12 < _S173)
                {
                }
                else
                {

#line 544
                    break;
                }

                if(i_12 < num_features_10)
                {

#line 548
                    float _S182 = ((input_10).load<float>((_S168), (uint(i_12)), (_S169), (_S170)));
                    float _S183 = ((input_10).load<float>((_S168), (uint(i_12)), (_S180), (_S181)));

#line 549
                    sample_val_5 = _S183;

#line 549
                    center_val_5 = _S182;

#line 547
                }
                else
                {


                    if(i_12 == num_features_10)
                    {

#line 552
                        i_13 = yoff_5;

#line 552
                    }
                    else
                    {

#line 552
                        i_13 = xoff_5;

#line 552
                    }

#line 552
                    sample_val_5 = float(i_13);

#line 552
                    center_val_5 = 0.0f;

#line 547
                }

#line 555
                float diff_7 = sample_val_5 - center_val_5;

                float _S184 = ((params_10).load<float>((_S168), (uint(i_12)), (_S169), (_S170)));

                float sample_weight_11 = sample_weight_10 * (F32_exp((_S184 * diff_7 * diff_7)));

#line 544
                i_12 = i_12 + int(1);

#line 544
                sample_weight_10 = sample_weight_11;

#line 544
            }

#line 544
            i_13 = int(0);

#line 544
            float dLdSampleWeight_4 = dLdTotalWeight_4;

#line 563
            for(;;)
            {

#line 563
                if(i_13 < num_features_10)
                {
                }
                else
                {

#line 563
                    break;
                }

#line 564
                float _S185 = ((output_grad_4).load<float>((_S168), (uint(i_13)), (_S169), (_S170)));

#line 564
                float _S186 = ((input_10).load<float>((_S168), (uint(i_13)), (_S180), (_S181)));

#line 564
                float dLdSampleWeight_5 = dLdSampleWeight_4 + _S185 * _S186 / _S167;

#line 563
                i_13 = i_13 + int(1);

#line 563
                dLdSampleWeight_4 = dLdSampleWeight_5;

#line 563
            }

#line 563
            int i_14 = int(0);

#line 568
            for(;;)
            {

#line 568
                if(i_14 < _S173)
                {
                }
                else
                {

#line 568
                    break;
                }

                if(i_14 < num_features_10)
                {

#line 572
                    float _S187 = ((input_10).load<float>((_S168), (uint(i_14)), (_S169), (_S170)));
                    float _S188 = ((input_10).load<float>((_S168), (uint(i_14)), (_S180), (_S181)));

#line 573
                    sample_val_5 = _S188;

#line 573
                    center_val_5 = _S187;

#line 571
                }
                else
                {

#line 571
                    int _S189;

#line 576
                    if(i_14 == num_features_10)
                    {

#line 576
                        _S189 = yoff_5;

#line 576
                    }
                    else
                    {

#line 576
                        _S189 = xoff_5;

#line 576
                    }

#line 576
                    sample_val_5 = float(_S189);

#line 576
                    center_val_5 = 0.0f;

#line 571
                }

#line 580
                float junk_5;

                float diff_8 = sample_val_5 - center_val_5;

                float dLdParam_0 = dLdSampleWeight_4 * sample_weight_10 * diff_8 * diff_8;
                uint _S190 = uint(i_14);

#line 585
                float _S191 = ((params_grad_4).load<float>((_S168), (_S190), (_S169), (_S170)));

#line 585
                (params_grad_4).store<float>((_S168), (_S190), (_S169), (_S170), (_S191 + dLdParam_0));

                if(i_14 < num_features_10)
                {
                    float _S192 = dLdSampleWeight_4 * sample_weight_10;

#line 589
                    float _S193 = ((params_10).load<float>((_S168), (uint(i_14)), (_S169), (_S170)));

#line 589
                    float dLdCenterVal_2 = _S192 * _S193 * -2.0f * diff_8;
                    int4  _S194 = make_int4 (b_10, i_14, y_11, x_11);

#line 590
                    uint4  _S195 = make_uint4 ((uint)_S194.x, (uint)_S194.y, (uint)_S194.z, (uint)_S194.w);

#line 590
                    *((&junk_5)) = atomicAdd((input_grad_4).data_ptr_at<float>((_S195)), (dLdCenterVal_2));

                    float _S196 = ((output_grad_4).load<float>((_S168), (uint(i_14)), (_S169), (_S170)));
                    int4  _S197 = make_int4 (b_10, i_14, yp_5, xp_5);

#line 593
                    uint4  _S198 = make_uint4 ((uint)_S197.x, (uint)_S197.y, (uint)_S197.z, (uint)_S197.w);

#line 593
                    *((&junk_5)) = atomicAdd((input_grad_4).data_ptr_at<float>((_S198)), (_S196 * sample_weight_10 / _S167 - dLdCenterVal_2));

#line 587
                }

#line 568
                i_14 = i_14 + int(1);

#line 568
            }

#line 534
            xoff_5 = xoff_5 + int(1);

#line 534
        }

#line 533
        yoff_5 = yoff_5 + int(1);

#line 533
    }

#line 600
    return;
}


#line 698
__global__ void __kernel__bwd_pixel_bilateral_filter_wrapper(TensorView input_11, TensorView input_grad_5, TensorView params_11, TensorView params_grad_5, TensorView output_11, TensorView output_grad_5, int kernel_boundary_11, int dialation_11)
{

#line 605
    int globalIdx_5 = int(((threadIdx)).x + ((blockIdx)).x * ((blockDim)).x);
    uint _S199 = ((input_11).sizes[(0U)]);

#line 606
    int batch_size_5 = int(_S199);
    uint _S200 = ((input_11).sizes[(1U)]);

#line 607
    int num_features_11 = int(_S200);
    uint _S201 = ((input_11).sizes[(3U)]);

#line 608
    int width_11 = int(_S201);
    uint _S202 = ((input_11).sizes[(2U)]);

#line 609
    int height_11 = int(_S202);
    if(globalIdx_5 >= batch_size_5 * width_11 * height_11)
    {

#line 611
        return;
    }
    int x_12 = globalIdx_5 % width_11;
    int _S203 = globalIdx_5 / width_11;

#line 614
    int y_12 = _S203 % height_11;
    int b_11 = globalIdx_5 / (width_11 * height_11);

#line 710
    bwd_pixel_bilteral_filter_0(num_features_11, width_11, height_11, b_11, y_12, x_12, input_11, input_grad_5, params_11, params_grad_5, output_11, output_grad_5, kernel_boundary_11, dialation_11);
    return;
}

