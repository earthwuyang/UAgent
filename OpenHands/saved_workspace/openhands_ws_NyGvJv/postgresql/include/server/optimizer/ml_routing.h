/*-------------------------------------------------------------------------
 *
 * ml_routing.h
 *    Declarations for machine learning-based query routing
 *
 * IDENTIFICATION
 *    src/include/optimizer/ml_routing.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_ROUTING_H
#define ML_ROUTING_H

#include "nodes/parsenodes.h"

/* GUC variables */
extern bool ml_routing_enabled;
extern int ml_routing_threshold;
extern char *ml_routing_model_path;

/* Feature structure */
typedef struct QueryFeatures {
    int command_type;           /* CMD_SELECT, CMD_INSERT, etc. */
    int num_tables;             /* Number of tables in query */
    int num_joins;              /* Number of joins */
    int num_aggregates;         /* Number of aggregate functions */
    int num_subqueries;         /* Number of subqueries */
    int num_groupby;            /* Number of GROUP BY clauses */
    int num_orderby;            /* Number of ORDER BY clauses */
    int num_where_clauses;      /* Number of WHERE clauses */
    int query_length;           /* Length of query string */
    bool has_window_funcs;      /* Has window functions */
    bool has_distinct;          /* Has DISTINCT */
    bool has_recursive_cte;     /* Has recursive CTE */
    bool has_modifying_cte;     /* Has modifying CTE */
    double estimated_cost;      /* PostgreSQL estimated cost */
} QueryFeatures;

/* Function declarations */
extern void InitializeMLRoutingGUC(void);
extern bool ml_route_query(Query *query, const char *query_text);
extern void collect_dual_execution_data(Query *query, const char *query_text);
extern bool ml_routing_hook(Query *query, const char *query_text);

/* C wrapper function that can be called from C++ code */
extern bool ml_routing_hook_wrapper(Query *query, const char *query_text);

#endif   /* ML_ROUTING_H */