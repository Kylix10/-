class ParsedRangeQuery:
    def __init__(self, query, db, table, col_left, col_right):
        self.query = query
        self.db = db
        self.table = table
        self.col_left = col_left
        self.col_right = col_right

    def get_tables(self):
        return [self.db + '.' + self.table]

    def get_join_conditions(self):
        # 这里假设连接条件是在where子句中包含'join'关键字的条件
        conditions = []
        if 'join' in self.query:
            where_index = self.query.find('where')
            if where_index != -1:
                where_clause = self.query[where_index + len('where'):]
                and_conditions = where_clause.split('and')
                for condition in and_conditions:
                    if 'join' in condition:
                        conditions.append(condition.strip())
        return conditions

    def get_predicate_conditions(self):
        # 假设谓词条件是在where子句中不包含'join'关键字的条件
        conditions = []
        if 'where' in self.query:
            where_index = self.query.find('where')
            where_clause = self.query[where_index + len('where'):]
            and_conditions = where_clause.split('and')
            for condition in and_conditions:
                if 'join' not in condition:
                    conditions.append(condition.strip())
        return conditions

    def column_range(self, col_name, min_val, max_val):
        """
        column_range returns the range of the specified column in this query.
        """
        if col_name not in self.col_left:
            return min_val, max_val
        return self.col_left[col_name], self.col_right[col_name]

    def column_names(self):
        """
        column_names returns column names that appear in this query.
        """
        cols = []
        for col in self.col_left:
            cols.append(col)
        return cols

    @staticmethod
    def parse_range_query(query):
        """
        parse_range_query parses a well formatted range query and return a ParsedRangeQuery.
        A well formatted range query looks like 'select * from db.t where c1>? and c1<? and c2>? and c2<? ...'.
        And types of all columns that appear in the query are supposed to be INT.
        """
        query = query.strip().lower()
        l = query.find("from")
        r = query.find("where")
        tmp = query[l+len("from"):r].strip().split(".")
        db, table = tmp[0], tmp[1]
        conds = query[r+len("where"):].split("and")
        col_left, col_right = {}, {}
        for cond in conds:
            cond = cond.strip()
            for op in ["<", ">"]:
                idx = cond.find(op)
                if idx > 0:
                    col = cond[:idx]
                    val = cond[idx+len(op):]
                    if op == ">":
                        col_left[col] = int(val)
                    elif op == "<":
                        col_right[col] = int(val)
        return ParsedRangeQuery(query, db, table, col_left, col_right)

    def __repr__(self):
        return f"RangeQuery{{db:{self.db}, table:{self.table}, col_left:{self.col_left}, col_right:{self.col_right}}}"