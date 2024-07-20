# Implementing a DBCopilot with LangChain

Architecture and implementation steps behind a DBCopilot application, a natural language interface to chat with database-structured data, leveraging LangChain's SQL Agent component

## LangChain agents and SQL Agent

LangChain agents are entities that drive decision making within LLM-powered applications. Agents have access to a suite of tools and can decide which tool to call based on the user input and the context. Agents are dynamic and adaptive, meaning that they can change or adjust their actions based on the situation or the goal.

`create_sql_agent`: An agent designed to interact with relational databases\
 `SQLDatabaseToolkit`: A toolkit to provide the agent with the required non-parametric knowledge\
 `OpenAI`: An LLM to act as the reasoning engine behind the agent, as well as the generative engine to produce conversational results\

### Implementation steps:

#### 1. Initialize all the components and establish the connection to the database, using the SQLDatabase LangChain component (which uses SQLAlchemy under the hood and is used to connect to SQL databases):

```python
from langchain.agents import create_sql_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
llm = OpenAI()
db = SQLDatabase.from_uri('sqlite:///chinook.db')
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
```

Included / available tools: `['sql_db_query', 'sql_db_schema', 'sql_db_list_tables', 'sql_db_query_checker']`\

Those tools have the following capabilities:

- `sql_db_query`: This takes as input a detailed and correct SQL query, and it outputs a result from the database. If the query is not correct, an error message will be returned.\
- `sql_db_schema`: This takes as input a comma-separated list of tables, and it outputs the schema and sample rows for those tables.\
- `sql_db_list_tables`: This takes as input an empty string, and it outputs a comma-separated list of tables in the database.\
- `sql_db_query_checker`: This tool double-checks whether the query is correct before executing it.

#### 2. Execute agent with a simple query to describe a table:

```
agent_executor.run("Describe the [TABLE] table")
```

This produces the following output (truncated):

```
> Entering new AgentExecutor chain... Action: sql_db_list_tables
Action Input:
Observation: ...
Thought: The table I need is [TABLE]
Action: sql_db_schema
Action Input: [TABLE]
Observation:
CREATE TABLE [TABLE] (
[...]
> Finished chain.
'The [TABLE] table contains the [COLUMNS] columns. It has a primary key of [PRIMARY_KEY]
```

As you can see, with a simple question in natural language, the agent is able to understand its semantics, translate it into an SQL query, extract the relevant information, and use it as context to generate the response.

It's able to do all of this because under the hood, the SQL agent comes with a default prompt template, which makes it tailored to this type of activity.

```
agent_executor.agent.llm_chain.prompt.template
```

The default template of the LangChain component:

```
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
If the question does not seem related to the database, just return "I don't know" as the answer.
sql_db_query: Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', using sql_db_schema to query the correct table fields.
sql_db_schema: Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: 'table1, table2, table3'
sql_db_list_tables: Input is an empty string, output is a comma separated list of tables in the database.
sql_db_query_checker: Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker]
Action Input: the input to the action
...
Question: {input}
Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
{agent_scratchpad}

```

This prompt template enables the agent to use the proper tools and generate a SQL query, without modifying the underlying database (you can see the explicit rule not to run any data manipulation language (DML) statements).

> DML is a class of SQL statements that are used to query, edit, add, and delete row-level data from database tables or views. DML statements are used to store, modify, retrieve, delete, and update data in a database. The main DML statements are as follows:
>
> - `SELECT`: This is used to retrieve data from one or more tables or views based on specified criteria.
> - `INSERT`: This is used to insert new data records or rows into a table.
> - `UPDATE`: This is used to modify the values of existing data records or rows in a table.
> - `DELETE`: This is used to remove one or more data records or rows from a table.
> - `MERGE`: This is used to combine the data from two tables into one based on a common column.

The agent is also able to correlate more than one table within the database:

```
agent_executor.run("What is the total number of [OBJECT 1] and the average value of [OBJECT 1] by [OBJECT 2]?")
```

Action Input then invokes two tables – [OBJECT 1] and [OBJECT 2]:

```
> Entering new AgentExecutor chain...
Action: sql_db_list_tables
Action Input:
Observation: ....
Thought: I should look at the schema of the [OBJECT 1] and [OBJECT 2] tables.
Action: sql_db_schema
Action Input: [OBJECT 1], [OBJECT 2]
[...]
```

The SQL query that the agent ran against the database can be printed to double check results.
To do so, modify the default prompt to ask the agent to explicitly show the reasoning behind its result.

## Prompt engineering

Default prompts can be customized and passed as a parameter to the component.
For example, let’s say that you want your SQL agent to print the SQL query it used to return the result.

The Agent takes a prompt `prefix` and a `format_instruction`, which are merged and constitute the default prompt.
To make your agent more self-explanatory, you can create two variables, `prefix` and `format_instructions`, which can be passed as parameters to modify the default prompt.

### Example

````Python

prefix = '''
You are an agent designed to interact with a SQL database.\nGiven an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.\nUnless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.\nYou can order the results by a relevant column to return the most interesting examples in the database.\nNever query for all the columns from a specific table, only ask for the relevant columns given the question.\nYou have access to tools for interacting with the database.\nOnly use the below tools. Only use the information returned by the below tools to construct your final answer.\nYou MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.\n\nDO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.\n\nIf the question does not seem related to the database, just return "I don\'t know" as the answer. As part of your final answer, ALWAYS include an explanation of how you got to the final answer, including the SQL query you run. Include the explanation and the SQL query in the section that starts with "Explanation:”
'''

prompt_format_instructions = """
Explanation:
<===Beginning of an Example of Explanation:
I joined the invoices and customers tables on the customer_id column, which is the common key between them. This will allowed me to access the Total and Country columns from both tables. Then I grouped the records by the country column and calculate the sum of the Total column for each country, ordered them in descending order and limited the SELECT to the top 5.
```sql
SELECT c.country AS Country, SUM(i.total) AS Sales
FROM customer c
JOIN invoice i ON c.customer_id = i.customer_id
GROUP BY Country
ORDER BY Sales DESC
LIMIT 5;
```sql
===>End of an Example of Explanation
"""

agent_executor = create_sql_agent(
  prefix=prompt_prefix,
  format_instructions=prompt_format_instructions,
  llm=llm,
  toolkit=toolkit,
  verbose=True,
  top_k=10
)
result = agent_executor.run("What are the top 5 best-selling albums and their artists?")
print(result)
````

### Example Output

````
The top 5 best-selling albums and their artists are 'A Matter of Life and Death' by Iron Maiden, 'BBC Sessions [Disc 1] [live]' by Led Zeppelin, 'MK III The Final Concerts [Disc 1]' by Deep Purple, 'Garage Inc. (Disc 1)' by Metallica and 'Achtung Baby' by U2.
Explanation: I joined the album and invoice tables on the album_id column and joined the album and artist tables on the artist_id column. This allowed me to access the title and artist columns from the album table and the total column from the invoice table. Then I grouped the records by the artist column and calculated the sum of the Total column for each artist, ordered them in descending order and limited the SELECT to the top 5.
```sql
SELECT al.title AS Album, ar.name AS Artist, SUM(i.total) AS Sales
FROM album al
JOIN invoice i ON al.album_id = i.invoice_id
JOIN artist ar ON al.artist_id = ar.artist_id
GROUP BY ar.name
ORDER BY Sales
````

## Adding further tools
