# Benchmarks

Information only: there is no speed bar. `.github/workflows/bench.yml` runs
`tests/parity/bench.sh` in the parity image. dav1d and av1-rs (`examples/oracle.rs`) decode the
same frames of a corpus subset (`quantizer-(00|31|63)`, `allintra`, `size-(196|226)`), once with
one thread and once with the default threads. The figure is the wall time of the whole set, best
of 3. av1-rs also writes its `.yuv` output; dav1d writes to null, so the ratio favours dav1d.

The result is the job summary of the `bench` run (`summary.md`: dav1d ms, av1-rs ms and the ratio
for 1 thread and default threads). A decoder command that exits non-zero is listed in the summary
as a warning, and the times of that run mean nothing. No numbers are copied here: the decoder is
scalar and spec-literal, and the job summary of the latest run on `main` is the record.
