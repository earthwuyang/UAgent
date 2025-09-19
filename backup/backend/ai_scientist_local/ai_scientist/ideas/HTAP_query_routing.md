# Title: HTAP query routing

## Keywords
row engine, column engine, query routing

## TL;DR
please use postgres and pg_duckdb extension to do HTAP query routing experiment, use machine learning classifiy a query will run faster on postgres or duckdb and route the query appropriately

## Abstract
please use postgres and pg_duckdb extension to do HTAP query routing experiment, use machine learning classifiy a query will run faster on postgres or duckdb and route the query appropriately.
You can expose features inside postgres kernel by modifying kernel code and then train a lightgbm model and then embed the lightgbm model inside postgres kernel and do the routing based on the prediction of lightgbm.