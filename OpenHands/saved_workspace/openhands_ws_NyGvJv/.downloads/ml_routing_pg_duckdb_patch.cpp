/*
 * ML-based query routing patch for pg_duckdb
 * 
 * This patch modifies pg_duckdb to integrate machine learning-based query routing
 * that decides whether to execute a query with PostgreSQL or DuckDB based on 
 * extracted query features.
 */

#include "pgduckdb/pgduckdb_hooks.hpp"
#include "pgduckdb/pgduckdb_utils.hpp"

/* Forward declaration of our ML routing function */
extern "C" bool ml_routing_hook(Query *query, const char *query_string);

namespace pgduckdb {

static PlannedStmt *
DuckdbPlannerHook_Cpp(Query *parse, const char *query_string, int cursor_options, ParamListInfo bound_params) {
	if (pgduckdb::IsExtensionRegistered()) {
		/* Call our ML routing hook first */
		bool should_route_to_duckdb = ml_routing_hook(parse, query_string);
		
		/* If ML routing says to use DuckDB, force it */
		if (should_route_to_duckdb) {
			pgduckdb::TriggerActivity();
			pgduckdb::IsAllowedStatement(parse, true);
			return DuckdbPlanNode(parse, cursor_options, true);
		}
		
		/* Otherwise, proceed with normal pg_duckdb logic */
		if (pgduckdb::NeedsDuckdbExecution(parse)) {
			pgduckdb::TriggerActivity();
			pgduckdb::IsAllowedStatement(parse, true);

			return DuckdbPlanNode(parse, cursor_options, true);
		} else if (pgduckdb::ShouldTryToUseDuckdbExecution(parse)) {
			pgduckdb::TriggerActivity();
			PlannedStmt *duckdbPlan = DuckdbPlanNode(parse, cursor_options, false);
			if (duckdbPlan) {
				return duckdbPlan;
			}
			/* If we can't create a plan, we'll fall back to Postgres */
		}
		if (parse->commandType != CMD_SELECT && !pgduckdb::pg::AllowWrites()) {
			elog(ERROR, "Writing to DuckDB and Postgres tables in the same transaction block is not supported");
		}
	}

	/*
	 * If we're executing a PG query, then if we'll execute a DuckDB
	 * later in the same transaction that means that DuckDB query was
	 * not executed at the top level, but internally by that PG query.
	 * A common case where this happens is a plpgsql function that
	 * executes a DuckDB query.
	 */

	pgduckdb::MarkStatementNotTopLevel();

	return prev_planner_hook(parse, query_string, cursor_options, bound_params);
}

} // namespace pgduckdb