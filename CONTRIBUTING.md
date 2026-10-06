# Contributing

Issues and pull requests are welcome.

- **Keep the paper and the code in step.** A change to a model that moves a published number should regenerate the result files (see the README) and say which numbers moved.
- **Tag every new input** in the code as `[REAL]` (with its source), `[HIST]` or `[ASSUMED]`, as the existing code does.
- **Tests.** `python -m unittest test_ai_bust_live` must pass. Add a test for any new parser.
- **Do not commit** the `live_history.db` database, raw downloaded data, credentials or server addresses.
- **Credit.** Contributions are licensed under Apache-2.0 (code) and CC BY 4.0 (text, results), the same as the project. By submitting one you confirm you have the right to do so.
- **Honest reporting.** Results from Model F ablations are noisy and loosely paired across seeds; please do not describe differences below the noise level as findings.
