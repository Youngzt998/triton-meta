# Retracted findings — ttgir-broad

These reports were written and committed before the line had a gate for
**undefined behaviour**, and every one of them is on a kernel that does a
*scatter store*: the write address is computed from loaded index data, so
two lanes of one launch can write the same address and the result depends
on the order the hardware runs them in. That is a write-write race, which
is UB, so two compilations are allowed to disagree (campaign plan section
4: data races are out of scope). They are not compiler bugs.

Retracted:

* `HIT-0001`
* `HIT-0002`
* `HIT-0003`
* `HIT-0004`
* `HIT-0005`
* `HIT-0008`
* `HIT-0016`
* `HIT-0020`

The gate that now catches this is `fz/irscan.py` (`ub_reason`), run in the
pre-screen and as a re-scan of kernels screened earlier. Kernels it flags
get status `ub-race` and are never swept.
