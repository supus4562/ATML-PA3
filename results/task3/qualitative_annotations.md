# Task 3: Qualitative Test-Time Compute & Adaptive Sampling Annotations

Representative examples illustrating selection disagreement, sequential refinement transitions, and adaptive stopping.

## 1. Live Refinement Transitions (MATH Subset)

| Sample ID | Initial Answer | Revised Answer | Transition Type |
|---|---|---|---|
| test:943 | `14` | `14` | Retained Correct |
| test:2688 | `-1` | `-1` | Retained Correct |
| test:3504 | `144` | `144` | Retained Wrong |
| test:4335 | `5040` | `5040` | Retained Correct |
| test:2460 | `NO_ANSWER` | `13` | Retained Wrong |
| test:1324 | `(1)/(2)` | `(1)/(2)` | Retained Wrong |
| test:2340 | `NO_ANSWER` | `NO_ANSWER` | Retained Wrong |
| test:1577 | `14` | `14` | Retained Correct |
| test:639 | `-9` | `-9` | Retained Correct |
| test:2892 | `23` | `23` | Retained Wrong |
| test:4702 | `1` | `1` | Retained Wrong |
| test:1709 | `62.5` | `62.5` | Retained Wrong |
| test:4041 | `56` | `56` | Retained Correct |
| test:2119 | `NO_ANSWER` | `21` | Retained Wrong |
| test:2087 | `7` | `7` | Retained Correct |

## 2. Selection Rule Disagreements (MathVista Subset)

| Question ID | FirstAns | Majority SC | Highest VisConf | RWC VisConf | Gold Answer |
|---|---|---|---|---|---|
| 99 | `d` | `d` | `d` | `d` | `D` |
| 425 | `a` | `a` | `a` | `a` | `A` |
| 601 | `frac13` | `frac13` | `frac13` | `frac13` | `0.33` |
| 77 | `2019` | `2019` | `2019` | `2019` | `2019` |
| 379 | `a` | `a` | `b` | `a` | `A` |
| 155 | `a` | `a` | `a` | `a` | `A` |
| 358 | `d` | `a` | `d` | `a` | `A` |
| 204 | `10` | `4` | `6` | `4` | `12` |
| 928 | `5` | `5` | `5` | `5` | `5` |
| 468 | `7` | `7` | `7` | `7` | `7` |

## 3. Adaptive Early Stopping Case Studies

Representative questions illustrating adaptive controller decisions (stopping budget $N \in \{4, 8, 16, 32\}$) vs. fixed-32 SC.

| Dataset | Question ID | Controller | Stopped $N$ | Selected Ans | Fixed-32 Ans | Gold Ans | Outcome Classification |
|---|---|---|---|---|---|---|---|
| MATH | test:943 | Quality-Aware | $N=4$ | `14` | `14` | `14` | Efficient Exit (Saved 88% compute) |
| MATH | test:2688 | Quality-Aware | $N=4$ | `-1` | `-1` | `-1` | Efficient Exit (Saved 88% compute) |
| MATH | test:3504 | Quality-Aware | $N=4$ | `216` | `216` | `216` | Efficient Exit (Saved 88% compute) |
| MATH | test:4335 | Quality-Aware | $N=4$ | `5040` | `5040` | `5040` | Efficient Exit (Saved 88% compute) |
| MATH | test:2460 | Quality-Aware | $N=4$ | `154` | `154` | `154` | Efficient Exit (Saved 88% compute) |
| MathVista | 99 | Quality-Aware | $N=4$ | `d` | `d` | `D` | Efficient Exit (Saved 88% compute) |
| MathVista | 425 | Quality-Aware | $N=4$ | `a` | `a` | `A` | Efficient Exit (Saved 88% compute) |
| MathVista | 601 | Quality-Aware | $N=4$ | `frac13` | `frac13` | `0.33` | Early Exit on Difficult Question |
| MathVista | 77 | Quality-Aware | $N=4$ | `2019` | `2019` | `2019` | Efficient Exit (Saved 88% compute) |
| MathVista | 379 | Quality-Aware | $N=4$ | `a` | `a` | `A` | Efficient Exit (Saved 88% compute) |
