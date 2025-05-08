import streamlit as st
import pandas as pd
import numpy as np
import pulp

st.markdown("""
**VBA Code Developed by:** Late B. N. Raviprakash  
**Translated & Enhanced by:** Manojkumar Patil  
**Description:** This Streamlit-based application generates Markov Chain equations and solves the defined LP problem.
""")

class MarkovLPSolver:
    def __init__(self):
        self.mcdata = None
        self.tot_rows = 0
        self.tot_cols = 0

    def calculate_markov(self, df, start_row, end_row, start_col, end_col):
        # Validate input parameters
        if not (1 <= start_row <= end_row <= len(df)):
            raise ValueError("Invalid row range. Please check your row selection.")
        if not (1 <= start_col <= end_col <= len(df.columns)):
            raise ValueError("Invalid column range. Please check your column selection.")

        start_row -= 1
        end_row -= 1
        start_col -= 1
        end_col -= 1
        
        self.tot_rows = end_row - start_row + 1
        self.tot_cols = end_col - start_col + 1
        self.mcdata = df.iloc[start_row:end_row+1, start_col:end_col+1].values
        
        # Validate row totals (last column)
        if np.any(self.mcdata[:, -1] == 0):
            raise ValueError("Row totals (last column) cannot contain zero values.")

        mcdata1 = self.mcdata.copy()
        
        for n in range(self.tot_cols - 1):
            for i in range(self.tot_rows):
                for j in range(self.tot_cols - 1):
                    mcdata1[i, j] = round((self.mcdata[i, j] / self.mcdata[i, -1]) * 10000) / 10000

        return mcdata1

    def solve_lp(self, transition_matrix):
        if transition_matrix.shape[0] < 2 or transition_matrix.shape[1] < 2:
            raise ValueError("Matrix dimensions too small. Need at least 2x2 matrix.")

        prob = pulp.LpProblem("Markov_LP_Problem", pulp.LpMinimize)
        
        variables = {}
        for i in range(self.tot_cols - 1):
            for j in range(self.tot_cols - 1):
                var_name = f"x{chr(97 + j)}{i+1}"
                variables[var_name] = pulp.LpVariable(var_name, lowBound=0)
        
        num_rows = min(transition_matrix.shape[0] - 1, self.tot_rows - 1)
        num_constraints = (self.tot_cols - 1) * num_rows
        u_vars = [pulp.LpVariable(f"u{i+1}", lowBound=0) for i in range(num_constraints)]
        v_vars = [pulp.LpVariable(f"v{i+1}", lowBound=0) for i in range(num_constraints)]
        
        prob += pulp.lpSum(u_vars) + pulp.lpSum(v_vars), "Objective_Function"
        
        row_num = 0
        for n in range(self.tot_cols - 1):
            for i in range(num_rows):
                constraint_expr = pulp.lpSum(
                    transition_matrix[i, j] * variables[f"x{chr(97 + j)}{n+1}"]
                    for j in range(self.tot_cols - 1)
                )
                target_value = transition_matrix[i+1, n] if i+1 < transition_matrix.shape[0] else 0
                prob += constraint_expr + u_vars[row_num] - v_vars[row_num] == target_value
                row_num += 1

        for n in range(self.tot_cols - 1):
            prob += pulp.lpSum(variables[f"x{chr(97 + n)}{j+1}"] for j in range(self.tot_cols - 1)) == 1
        
        try:
            prob.solve()
            if pulp.LpStatus[prob.status] != 'Optimal':
                raise ValueError(f"Could not find optimal solution. Status: {pulp.LpStatus[prob.status]}")
            results = {v.name: v.varValue for v in prob.variables()}
            return results, pulp.LpStatus[prob.status]
        except Exception as e:
            raise ValueError(f"Failed to solve LP problem: {str(e)}")


def main():
    st.title("MARKOV LP SOLVER")
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'xls', 'xlsm', 'csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())
        st.info(f"📊 Data Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        st.warning("⚠️ The last column in your selection must be the row total!")
        col_labels = {i+1: label for i, label in enumerate(df.columns)}
        st.write(col_labels)
        
        col1, col2 = st.columns(2)
        with col1:
            start_row = st.number_input("Start Row", min_value=1, value=1)
            end_row = st.number_input("End Row", min_value=start_row, value=len(df))
        with col2:
            start_col = st.number_input("Start Column", min_value=2, value=2)
            end_col = st.number_input("End Column (max = " + str(len(df.columns)) + ")", min_value=start_col, value=len(df.columns))
        
        if st.button("Solve LP Problem"):
            solver = MarkovLPSolver()
            try:
                transition_matrix = solver.calculate_markov(df, start_row, end_row, start_col, end_col)
                results, status = solver.solve_lp(transition_matrix)
                
                st.subheader("🎯 LP Solution")
                st.write(f"**Solution Status:** {status}")
                
                # Organize results in a matrix format
                matrix_data = {}
                for var_name, value in results.items():
                    if var_name.startswith('x'):
                        row = var_name[1]  # 'a', 'b', 'c', etc.
                        col = int(var_name[2])  # 1, 2, 3, etc.
                        if row not in matrix_data:
                            matrix_data[row] = {}
                        matrix_data[row][col] = value
                
                # Create a pandas DataFrame for better display
                rows = sorted(matrix_data.keys())
                max_col = max(col for row_data in matrix_data.values() for col in row_data.keys())
                df_display = pd.DataFrame(
                    [[matrix_data[row].get(i, 0) for i in range(1, max_col + 1)] for row in rows],
                    index=[f'x{row}' for row in rows],
                    columns=[str(i) for i in range(1, max_col + 1)]
                )
                df_display['Sum'] = df_display.sum(axis=1)
                
                # Display results in both styled table and raw format
                st.write("**Matrix Format with Row Sums**")
                st.dataframe(
                    df_display.style.format('{:.6f}')
                    .set_properties(**{
                        'text-align': 'center',
                        'font-family': 'monospace',
                        'width': '100px'
                    })
                    .set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center')]},
                        {'selector': '', 'props': [('border', '1px solid')]}
                    ])
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
