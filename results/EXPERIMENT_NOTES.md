# Experiment notes

## Final choice

The final model is:

```text
fastText -> dynamic shortlist-5 -> base GLiNER2
```

## Why LoRA was rejected

The adapter was trained on 221,438 automatically generated weak labels. It reduced:

- Span F1: 0.7035 -> 0.6596
- Attribute F1: 0.7674 -> 0.6974

The most likely causes are label noise, ambiguous numeric spans, and a mismatch between automatically labeled titles/descriptions and the final short-query distribution.

The adapter is not used by default. The experiment is preserved because it is an important engineering result: more training data and a more complex model did not improve the evaluated system.
