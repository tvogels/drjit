/*
    drjit/call.h -- Vectorized method call support. This header file provides
    the logic to capture a call to ``Jit/DiffArray<T*>().foo()`` and dispatch
    it to ``T::foo()``.

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/autodiff.h>
#include <drjit/struct.h>

#define DRJIT_CALL_BEGIN(Name)                                                 \
    namespace drjit {                                                          \
        template <typename Self>                                               \
        struct call_support<Name, Self> {                                      \
            using Class = Name;                                                \
            using Mask = mask_t<Self>;                                         \
            static constexpr const char *Domain = #Name;                       \
            call_support(const Self &self) : self(self) { }                    \
            const call_support *operator->() const {                           \
                return this;                                                   \
            }

#define DRJIT_CALL_TEMPLATE_BEGIN(Name)                                        \
    namespace drjit {                                                          \
        template <typename Self, typename... Ts>                               \
        struct call_support<Name<Ts...>, Self> {                               \
            using Class = Name<Ts...>;                                         \
            using Mask = mask_t<Self>;                                         \
            static constexpr const char *Domain = #Name;                       \
            call_support(const Self &self) : self(self) { }                    \
            const call_support *operator->() const {                           \
                return this;                                                   \
            }

#define DRJIT_CALL_END(Name)                                                   \
        private:                                                               \
            const Self &self;                                                  \
        };                                                                     \
    }

#define DRJIT_CALL_METHOD(Name)                                                \
public:                                                                        \
    template <typename... Args> auto Name(const Args &...args) const {         \
        return drjit_impl_##Name(std::make_index_sequence<sizeof...(Args)>(),  \
                                 args...);                                     \
    }                                                                          \
                                                                               \
private:                                                                       \
    template <typename... Args, size_t... Is>                                  \
    auto drjit_impl_##Name(std::index_sequence<Is...>, const Args &...args)    \
        const {                                                                \
        using Ret = decltype(std::declval<Class &>().Name(args...));           \
        using Ret2 = std::conditional_t<std::is_void_v<Ret>, std::nullptr_t,   \
                                        vectorize_t<Self, Ret>>;               \
        using CallStateT = detail::CallState<Ret2, Args...>;                   \
                                                                               \
        ad_call_func callback = [](void *state_p, void *self,                  \
                                   const dr_vector<uint64_t> &args_i,          \
                                   dr_vector<uint64_t> &rv_i) {                \
            CallStateT *state = (CallStateT *) state_p;                        \
            state->update_args(args_i);                                        \
            if constexpr (std::is_same_v<Ret, void>) {                         \
                if (self)                                                      \
                    ((Class *) self)->Name(state->args.template get<Is>()...); \
            } else {                                                           \
                if (self)                                                      \
                    state->rv = ((Class *) self)                               \
                                    ->Name(state->args.template get<Is>()...); \
                else                                                           \
                    state->rv = zeros<Ret2>();                                 \
                state->collect_rv(rv_i);                                       \
            }                                                                  \
        };                                                                     \
                                                                               \
        return detail::call<Self, Ret, Ret2, Args...>(                         \
            self, Domain, #Name "()", false, callback, args...);               \
    }

#define DRJIT_CALL_GETTER(Name)                                                \
public:                                                                        \
    auto Name(Mask mask = true) const {                                        \
        using Ret =                                                            \
            vectorize_t<Self, decltype(std::declval<Class &>().Name())>;       \
        using CallStateT = detail::CallState<Ret, Mask>;                       \
                                                                               \
        ad_call_func callback = [](void *state_p, void *self,                  \
                                   const dr_vector<uint64_t> &,                \
                                   dr_vector<uint64_t> &rv_i) {                \
            CallStateT *state = (CallStateT *) state_p;                        \
            if (self)                                                          \
                state->rv = ((Class *) self)->Name();                          \
            else                                                               \
                state->rv = zeros<Ret>();                                      \
            state->collect_rv(rv_i);                                           \
        };                                                                     \
                                                                               \
        return detail::call<Self, Ret, Ret, Mask>(self, Domain, #Name "()",    \
                                                  true, callback, mask);       \
    }
NAMESPACE_BEGIN(drjit)

template <typename Guide, typename T>
using vectorize_t =
    std::conditional_t<std::is_scalar_v<T>, replace_scalar_t<Guide, T>, T>;

template <bool IncRef, typename T>
void collect_indices(const T &value, dr_vector<uint64_t> &indices) {
    if constexpr (depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            collect_indices<IncRef>(value.derived().entry(i), indices);
    } else if constexpr (is_tensor_v<T>) {
        collect_indices<IncRef>(value.array(), indices);
    } else if constexpr (is_jit_v<T>) {
        uint64_t index = value.index_combined();
        if constexpr (IncRef)
            ad_var_inc_ref(index);
        indices.push_back(index);
    } else if constexpr (is_drjit_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](const auto &x) { collect_indices<IncRef>(x, indices); });
    }
}

template <typename T>
void update_indices(T &value, const dr_vector<uint64_t> &indices, size_t &pos) {
    if constexpr (depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            update_indices(value.derived().entry(i), indices, pos);
    } else if constexpr (is_tensor_v<T>) {
        update_indices(value.array(), indices, pos);
    } else if constexpr (is_jit_v<T>) {
        value = T::borrow((typename T::Index) indices[pos++]);
    } else if constexpr (is_drjit_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](auto &x) { update_indices(x, indices, pos); });
    }
}

template <typename T> void update_indices(T &value, const dr_vector<uint64_t> &indices) {
    size_t pos = 0;
    update_indices(value, indices, pos);
#if !defined(NDEBUG)
    if (pos != indices.size())
        throw std::runtime_error("update_indices(): did not consume the expected number of indices!");
#endif
}

NAMESPACE_BEGIN(detail)

template <typename Mask, typename ... Args>
Mask extract_mask(dr_tuple<Args...> &t) {
    constexpr size_t N = sizeof...(Args);
    Mask result = true;

    if constexpr (N > 0) {
        auto &last = t.template get<N-1>();
        if constexpr (is_mask_v<decltype(last)>)
            std::swap(result, last);
    }

    return result;
}

template <typename Ret, typename... Args> struct CallState {
    dr_tuple<Args...> args;
    Ret rv;

    CallState(const Args &...arg) : args(arg...) { }

    static void cleanup(void *p) {
        delete (CallState *) p;
    }

    void update_args(const dr_vector<uint64_t> &indices) {
        update_indices(args, indices);
    }

    void collect_rv(dr_vector<uint64_t> &indices) const {
        collect_indices<false>(rv, indices);
    }
};

struct dr_index_vector : dr_vector<uint64_t> {
    using Base = dr_vector<uint64_t>;
    using Base::Base;
    ~dr_index_vector() {
        for (size_t i = 0; i < size(); ++i)
            ad_var_dec_ref(operator[](i));
    }
};

template <typename Self, typename Ret, typename Ret2, typename... Args>
Ret call(const Self &self, const char *domain, const char *name,
         bool is_getter, ad_call_func callback, const Args &...args) {
    using Mask = mask_t<Self>;
    using CallStateT = CallState<Ret2, Args...>;
    CallStateT *state = new CallStateT(args...);

    Mask mask = extract_mask<Mask>(state->args);

    dr_index_vector args_i, rv_i;
    collect_indices<true>(state->args, args_i);
    bool done = ad_call(Self::Backend, domain, 0, name, is_getter,
                         self.index(), mask.index(), args_i, rv_i, state,
                         callback, &CallStateT::cleanup, true);

    if constexpr (!std::is_same_v<Ret, void>) {
        Ret2 result(std::move(state->rv));
        update_indices(result, rv_i);

        if (done)
            CallStateT::cleanup(state);

        return result;
    } else {
        if (done)
            CallStateT::cleanup(state);
    }
}

NAMESPACE_END(detail)
NAMESPACE_END(drjit)