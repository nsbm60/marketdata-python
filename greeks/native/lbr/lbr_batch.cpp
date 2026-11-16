//
// Batch entry point over Peter Jäckel's reference "Let's Be Rational" implementation (vendored in
// this directory; see LICENSE — Jäckel's permissive notice, preserved). This file is ours; the
// algorithm and every other .cpp/.h here are Jäckel's, unmodified.
//
// One function: invert a batch of undiscounted Black prices to implied Black volatilities. The caller
// (greeks.solver.invert_batch) has already applied every pre-check as an array mask and passes only
// the rows still live, with `price` already undiscounted — the same quantity py_vollib passes to the
// routine today. Where Jäckel signals no solution (±DBL_MAX for below-intrinsic / above-maximum) we
// write NaN, which the Python side maps to SOLVER_FAILED exactly as the ±FLOAT_MAX sentinels do now.
//
#include "lets_be_rational.h"

#include <cstddef>
#include <float.h>
#include <limits>

extern "C" DLL_EXPORT int lbr_implied_vol_batch(
        size_t n,
        const double* price,
        const double* forward,
        const double* strike,
        const double* t,
        const int* is_call,
        double* out_iv) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < n; ++i) {
        const double q = is_call[i] ? 1.0 : -1.0;
        const double iv = ImpliedBlackVolatility(price[i], forward[i], strike[i], t[i], q);
        out_iv[i] = (iv == DBL_MAX || iv == -DBL_MAX) ? nan : iv;
    }
    return 0;
}
