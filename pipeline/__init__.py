"""Article preprocessing pipeline: clean -> flatten -> chunk -> embed.

Each phase reads and writes parquet files so the phases stay independent
(deployable as separate SageMaker pipeline steps later). All tunable
constants live in pipeline.config.
"""