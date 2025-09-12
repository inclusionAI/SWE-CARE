### Code Review Annotation Questionnaire

**Instance ID**: `dbt-labs__dbt-core-3100@4f8c10c`  
**Repository**: `dbt-labs/dbt-core`  
**Pull Request**: [`#3100`](https://github.com/dbt-labs/dbt-core/pull/3100)  
**Language**: `Python`  
**Created at**: `2021-02-13T16:24:06Z`  
**Base commit**: `934c23bf392c4d5db9b129d0f8953c2b7872f5b0`  
**Commit to review**: `4f8c10c1aab3630f175611e87903b6051afdd8d8`
**Commit message**:
> default to get_columns_in_relation if not specified in config
> 
> Co-authored-by: Jeremy Cohen <jtcohen6@gmail.com>

---

### Context

#### Problem Statement

> Specify Columns to update on incremental Models
> ### Describe the feature
> Allow model developer the ability to choose the set of columns that require updating on incremental loads
> 
> ### Who will this benefit?
> This will benefit all users that are performing an incremental update of a wide table with few columns that are mutable

#### Hints (pre-PR comments or linked issues context)

> hey @richmintz! Can you say more about this? I think it's a good idea - I'm curious if you're interested in this feature for performance reasons, or something else.
> Hi @drewbanin!
> My primary goal would be performance, minimizing writes and index rebuilds. However I also think it will generate more concise sql for large tables when a merge is required. 
> Got it! Sure, this makes a lot of sense in `merge` statements. Is this on Snowflake/BigQuery? I think a similar paradigm would apply to Redshift/Postgres, but we'd modify the `update` statement instead of the `merge` statement.
> 
> I can imagine a config like `update_columns` which could be supplied like:
> ```
> {{
>   config(
>     materialized='incremental',
>     update_columns=['order_total', 'count_orders']
>   )
> }}
> ```
> 
> If `update_columns` is not provided, or if it is `null`, dbt will default to updating all of the columns in the model.
> 
> You buy all of that?
>  
> 
> As part of this proposed change, it would also be nice to be able to exclude the `when matched then update set` part of the merge altogether, as in some of my models I'm only interested in adding new rows since the source data is never updated (for event-based data for example or other append-only tables), and it makes the model execution faster (at least in BigQuery).
> 
> It could be a separate config `no_update = True` or just a convention that doesn't include the SQL block when `update_columns` is empty.
> 
> Please note I tested this and would work for BigQuery, other databases might need a different syntax to support no-op for updates.
> 
> Any thoughts?
> hey @bfil - if you do not supply a `unique_key` for your incremental model's config, then dbt should not inject a `when matched then update set....` clause to the merge statement. 
> 
> Check out the implementation here:
> https://github.com/fishtown-analytics/dbt/blob/7a07017b96ac332c872725343833a94b49129c68/core/dbt/include/global_project/macros/materializations/common/merge.sql#L23-L48
> @drewbanin I know, but I still want to compare and insert only the new data based on that unique key rather than merging on an always `FALSE` condition. Makes sense?
> ah! Sure, that makes a lot of sense. What do you think about pushing this code down from the materialization layer and into the modeling layer. Could you do something like:
> 
> ```
> {{ config(materialized='incremental') }}
> 
> select id, col1, col2 from {{ ref('some_table') }}
> {% if is_incremental() %}
> where id not in (select id from {{ this }})
> {% endif %}
> ```
> 
> This is a kind of funny adaptation of the [typical incremental modeling approach](https://docs.getdbt.com/docs/configuring-incremental-models), but it should serve to only select the _new_ records from your source table, inserting them directly into the destination table.
> @drewbanin : any update on where we are with this feature...i have a wide table [Snowflake] but i only need to update very few columns in the incremental model. any ETA's on the feature ?
> hey @ee07dazn - no movement on this issue on our end to report. I think the spec I mentioned above (pasted below) is still a good idea:
> 
> ```
> {{
>   config(
>     materialized='incremental',
>     update_columns=['order_total', 'count_orders']
>   )
> }
> ```
> 
> I think that this feature should only be supported on databases that support `merge` statements, as dbt's delete+insert approach doesn't really lend itself well to this approach. If anyone is interested in working on this one, I'd be happy to try to point you in the right direction. 
> 
> The key change here will be to replace the call to [`adapter.get_columns_in_relation`](https://github.com/fishtown-analytics/dbt/blob/f3565f3f70062c5818395d03a52c89e87616f956/plugins/snowflake/dbt/include/snowflake/macros/materializations/incremental.sql#L59) with `config.get('update_columns')`. We'll want to implement this for both the Snowflake and BigQuery incremental materializations when the `merge` strategy is used.
> Thanks @drewbanin for the information and a pointer. Apart from what you had suggested, i had to make change to the default__get_merge_sql macro to make that approach work. It works now but i am just not happy to make a change to something as low level as default__get_merge_sql. Probably i think i can update the snowflake_get_merge_sql to do this job. Thinking out loud...but thanks for the help.

#### Resolved Issues




##### Issue #1862

- Title:

> Specify Columns to update on incremental Models

- Body:

> ### Describe the feature
> Allow model developer the ability to choose the set of columns that require updating on incremental loads
> 
> ### Who will this benefit?
> This will benefit all users that are performing an incremental update of a wide table with few columns that are mutable



#### Patch to Review (diff)

```diff
diff --git a/plugins/bigquery/dbt/include/bigquery/macros/materializations/incremental.sql b/plugins/bigquery/dbt/include/bigquery/macros/materializations/incremental.sql
index f4ad80d5a5c..a48178f88a3 100644
--- a/plugins/bigquery/dbt/include/bigquery/macros/materializations/incremental.sql
+++ b/plugins/bigquery/dbt/include/bigquery/macros/materializations/incremental.sql
@@ -112,7 +112,10 @@
       {% endif %}
       {% set build_sql = create_table_as(False, target_relation, sql) %}
   {% else %}
-     {% set dest_columns = adapter.get_columns_in_relation(existing_relation) %}
+     {% set dest_columns = config.get('update_columns', none) %}
+     {% if dest_columns is none %}
+         {% set dest_columns = adapter.get_columns_in_relation(existing_relation) %}
+     {% endif %}
 
      {#-- if partitioned, use BQ scripting to get the range of partition values to be updated --#}
      {% if strategy == 'insert_overwrite' %}
diff --git a/plugins/snowflake/dbt/include/snowflake/macros/materializations/incremental.sql b/plugins/snowflake/dbt/include/snowflake/macros/materializations/incremental.sql
index 188795b1537..6d0d3bf9805 100644
--- a/plugins/snowflake/dbt/include/snowflake/macros/materializations/incremental.sql
+++ b/plugins/snowflake/dbt/include/snowflake/macros/materializations/incremental.sql
@@ -58,7 +58,7 @@
     {% do adapter.expand_target_column_types(
            from_relation=tmp_relation,
            to_relation=target_relation) %}
-    {% set dest_columns = adapter.get_columns_in_relation(target_relation) %}
+    {% set dest_columns = config.get('update_columns') %}
     {% set build_sql = dbt_snowflake_get_incremental_sql(strategy, tmp_relation, target_relation, unique_key, dest_columns) %}
   {% endif %}
 

```

#### Reference Review Comments (from the PR)

Count: 1


---

##### Comment 1

- Review comment:
  
> @user1:
> As you did for BigQuery, this should be:
> ```suggestion
>     {% set dest_columns = config.get('update_columns', none) %}
>     {% if dest_columns is none %}
>         {% set dest_columns = adapter.get_columns_in_relation(existing_relation) %}
>     {% endif %}
> ```
> I believe this is what's causing the failing integration test:
> ```
> 'NoneType' object is not iterable
> ```
> 
> @author:
> of course 🤦

- Diff hunk the review comment applied to:

```diff
@@ -58,7 +58,7 @@
     {% do adapter.expand_target_column_types(
            from_relation=tmp_relation,
            to_relation=target_relation) %}
-    {% set dest_columns = adapter.get_columns_in_relation(target_relation) %}
+    {% set dest_columns = config.get('update_columns') %}
```

- Path: `plugins/snowflake/dbt/include/snowflake/macros/materializations/incremental.sql`
- Line: n/a  |  Start line: n/a
- Original line: 61  |  Original start line: n/a


---

### Section 1 — Problem Statement and Patch Alignment

- **Q1.1 — Is the problem statement sufficiently well-specified for reviewing this patch?**

  _Your notes:_

- **Q1.2 — Explain your choice above. Reference specific files/functions when relevant.**

  _Your notes:_

- **Q1.3 — Does the patch address the stated problem?**
- [ ] Yes
- [ ] Partially
- [ ] No

  - **Why?**

---

### Section 2 — Review Scope and Comment Coverage

- **Q2.1 — Do the provided reference review comments appropriately cover the patch (scope and correctness)?**

  _Your notes:_

- **Q2.2 — Explain your choice. Identify any mis-scoped or missing review topics.**

  _Your notes:_

- **Q2.3 — Reference Review Comments Assessment**

For each reference comment, mark whether it is a true positive (valid issue), false positive (not actually an issue), or informational. Briefly justify.

| Index | Category (Functionality/Correctness/Performance/Security/Maintainability/Style/Docs) | Verdict (TP/FP/Info) | Notes |
| --- | --- | --- | --- |
| 1 |   |   |   |

- **Q2.4 — Missing Review Points**

List important review findings not covered by the reference comments.

  - [ ] Item 1:
  - [ ] Item 2:
  - [ ] Item 3:

---

### Section 3 — Defects Identified in the Patch

Repeat the following block for each defect you identify.

```text
Defect N
- Category: [ ] Functionality  [ ] Correctness  [ ] Performance  [ ] Security  [ ] Maintainability  [ ] Style  [ ] Documentation
- Severity (1-5):
- Files/Locations:
- Short description:
- Suggested fix (optional):
```

---

### Section 4 — Difficulty and Review Effort

- **Q4.1 — Difficulty to understand and correctly review the patch**
- [ ] <15 min
- [ ] 15 min - 1 hour
- [ ] 1-4 hours
- [ ] >4 hours

  - **Why?**

  - Time estimate:

- **Q4.2 — Estimated review effort (1-5)**
- [ ] 1: Trivial, almost no cognitive load
- [ ] 2: Small effort, localized changes
- [ ] 3: Moderate effort, multiple files/concerns
- [ ] 4: High effort, complex interactions or risk
- [ ] 5: Very high effort, wide impact or intricate logic

  - **Rationale:**

  - Effort level:

---

### Section 5 — Overall Patch Quality and Risk

- **Q5.1 — Does the patch meet acceptance criteria for the stated problem?**
- [ ] Yes
- [ ] Yes, with minor issues
- [ ] No

  - **Notes:**

- **Q5.2 — Regression risk assessment**
- [ ] Low
- [ ] Medium
- [ ] High

  - **Why?**

  - Risk level:

- **Q5.3 — Additional observations (tests, docs, API changes, compatibility)**

  _Your notes:_

---

### Section 6 — Dataset Suitability

- **Q6.1 — Is this instance suitable for evaluating code review quality?**
- [ ] Yes
- [ ] No

  - If 'No', explain (e.g., ambiguous problem, non-diff artifacts, excessive binary/auto-generated changes, missing context):

- **Q6.2 — Any compliance, privacy, or licensing concerns?**
- [ ] No
- [ ] Yes

  - If 'Yes' explain:

---

### Section 7 — Confidence

- **Q7.1 — How confident are you in your annotations? (1 = very low, 5 = very high)**

  Select one: [ ] 1  [ ] 2  [ ] 3  [ ] 4  [ ] 5