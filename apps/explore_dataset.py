import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as npa
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def read_clean_data():
    return pd.read_csv('data/basketball_data_cleaned.csv')

def filter_data(ds, column, tab):
    float_cols = ds.select_dtypes(include="float").columns
    int_cols = ds.select_dtypes(include="int").columns
    categorical_cols = ds.select_dtypes(include="object").columns
    
    display_cols = ["Name", "Position", "Team", "Year"]
    display_cols = display_cols+[column] if column not in display_cols else display_cols

    if column in float_cols:
        min_val = ds[column].min()
        max_val = ds[column].max()
        q1 = np.quantile(ds[column], 0.25)
        q3 = np.quantile(ds[column], 0.75)
        
        values = tab.slider(f'See which players recorded `{column}` within the range provided', float(min_val),float(max_val), (float(q1), float(q3)))
        
        return ds.loc[ds[column].between(values[0], values[1]), display_cols]
    
    elif column in int_cols:
        min_val = ds[column].min()
        max_val = ds[column].max()
        values = tab.slider(f'See which players recorded `{column}` within the range provided', int(min_val), int(max_val), (int(min_val), int(max_val)), step=1)
        
        return ds.loc[ds[column].between(values[0], values[1]), display_cols]

    elif column in categorical_cols:
        options = ds[column].unique().tolist()
        selectbox = tab.multiselect(f"Select `{column}` values to filter", options)
        return ds.loc[ds[column].isin(selectbox), display_cols]


def app():
    df = read_clean_data()
    st.title("Basketball Data")
    
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Filter Data", "Univariate Analysis", "Bivariate Analysis", "Machine Learning", "NLP"])

    with tab1:
        df = read_clean_data()
        st.title("Basketball Data")
        column = st.selectbox("Choose a column", df.columns)
        filtered_df = filter_data(df, column, tab1)
        st.write(filtered_df)
        
    with tab2:
        st.write('Summary Statistics')
        st.write(df.drop(columns='Year').describe())
        
        
        column = st.selectbox("Choose a column", df.select_dtypes(include="float").columns)
        colour = st.color_picker('Pick A Colour', "#51FF00")
        
        permission_to_plot = False
        
        if st.button(f'Plot Boxplot of {column}'):
            fig = px.box(df, x=column, color_discrete_sequence=[colour])
            permission_to_plot = True
        
        if st.button(f'Plot Histogram of {column}'):
            fig = px.histogram(df, x=column, histnorm='probability density', color_discrete_sequence=[colour])
            permission_to_plot = True
            
        if st.button(f'Plot Cumulative Histogram of {column}'):
            fig = px.histogram(df, x=column, color_discrete_sequence=[colour], histnorm='probability density', cumulative=True)
            permission_to_plot = True
        
        if st.button(f'Plot Violin of {column}'):
            fig = px.violin(df, x=column, box=True, color_discrete_sequence=[colour], points='outliers')
            permission_to_plot = True
            
        if st.button(f'Plot ECDF of {column}'):
            fig = px.ecdf(df, x=column, color_discrete_sequence=[colour])
            permission_to_plot = True
           
        if permission_to_plot==True: 
            title = go.layout.Title(text=f"Univariate Analysis of {column}", x=0.5, xref='paper', xanchor='center')
            fig.update_layout(title=title)
            tab2.plotly_chart(fig)
        
    with tab3:
        permission_to_plot = False
        analysis = st.selectbox('How would you like to perform your analysis', ['Continuous vs Continuous'])

        if analysis == 'Continuous vs Continuous':
            if st.button('Correlation Heatmap'):
                corr = df.drop(columns='Year').corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                corr_plot = corr.mask(mask)

                heat = go.Heatmap(
                    z=corr_plot,
                    x=corr_plot.columns.values,
                    y=corr_plot.columns.values,
                    zmin=-1, # Sets the lower bound of the color domain
                    zmax=1,
                    xgap=1, # Sets the horizontal gap (in pixels) between bricks
                    ygap=1,
                    colorscale='viridis_r'
                )

                title = 'Correlation Matrix'

                height = width = 600  # make customisable
                layout = go.Layout(
                    title='Correlation Matrix', 
                    title_x=0.5,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    yaxis_autorange='reversed',
                    height=600, width=600
                )

                fig=go.Figure(data=[heat], layout=layout)
                permission_to_plot = True
        
            # Smort placement
            st.markdown("### We now turn our attention to the $xy$ Plane")
            
            colour = st.color_picker('Pick A Colour', "#51FF00", key='cp2')
            opt = df.drop(columns='Year').select_dtypes(include="float").columns.tolist()
            x = st.selectbox(label='x', options=opt)
            y = st.selectbox(label='y', options=opt)
            
            r = df[x].corr(df[y], method='pearson')
            
            if st.button('Generate Scatterplot'):
                trace = go.Scatter(x=df[x], y=df[y], mode='markers', marker=dict(color=colour))
                fig = go.Figure(data=[trace])
                fig.update_layout(
                    title=go.layout.Title(
                        text=f"{df[y].name} vs {df[x].name} (r = {np.round(r, 3)})",
                        x=0.5,
                        xref='paper',
                        xanchor='center'
                    ),
                )
                permission_to_plot = True
                
                
            if permission_to_plot==True: 
                tab3.plotly_chart(fig)
        
    with tab4:
        st.write('Machine Learning')
        option = st.selectbox('What would you like to do', ['Model Building'])
        if option == 'Model Building':
            num_cols = df.select_dtypes(include=np.number).columns
            Xcols = st.multiselect(
                'Select columns to place in your feature matrix',
                options=num_cols
            )
            ycol = st.selectbox('Select Response Vector',
                             options=[c for c in num_cols if c not in Xcols])
            
            if len(Xcols)>0 and len([ycol])==1:
                X = df[Xcols]
                y = df[ycol]
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
                if st.button('Build Linear Regression'):
                    with st.spinner('Training Linear Regression Model...'):
                        model = Ridge(alpha=0.5)
                        model.fit(X_train, y_train)
                        r2 = model.score(X_test, y_test)
                        mse = mean_squared_error(y_test, X_test)
                        mae = mean_absolute_error(y_test, X_test)
                        
                        st.success('Model Trained!')
                        st.markdown(f"$R^2$={np.round(r2, 3)}")
                        st.markdown(f"$MSE$={np.round(mse, 3)}")
                        st.markdown(f"$MAE$={np.round(mae, 3)}")

                    # # Set up the parameter grid for Ridge Regression
                    # param_grid = {'alpha': [0.1, 0.5, 1, 2, 5, 10]}

                    # # Use GridSearchCV for Ridge Regression
                    # grid_search_ridge = GridSearchCV(Ridge(), param_grid, cv=5)
                    # grid_search_ridge.fit(X_train, y_train)

                    # # Plot the mean squared error as alpha changes
                    # alphas = param_grid['alpha']
                    # mse_scores = []
                    # for alpha in alphas:
                    #     ridge = Ridge(alpha=alpha)
                    #     ridge.fit(X_train, y_train)
                    #     y_pred = ridge.predict(X_test)
                    #     mse = mean_squared_error(y_test, y_pred)
                    #     mse_scores.append(mse)

                    # st.line_chart(mse_scores)
                if st.button('Build Random Forest Regressor'):
                    with st.spinner('Training Random Forest Regression Model...'):
                        model = RandomForestRegressor().fit(X_train, y_train)
                        r2 = model.score(X_test, y_test)
                        mse = mean_squared_error(y_test, X_test)
                        mae = mean_absolute_error(y_test, X_test)
                        time.sleep(5)
                    st.success('Model Trained!')
                    st.markdown(f"$R^2$={np.round(r2, 3)}")
                    st.markdown(f"$MSE$={np.round(mse, 3)}")
                    st.markdown(f"$MAE$={np.round(mae, 3)}")
                if st.button('Build Neural Network'):
                    with st.spinner('Training Neural Network...'):
                        time.sleep(5)
                    st.success('Model!')
                    df = read_clean_data()
                    if len(Xcols)>0 and len([ycol])==1:
                        X = df[Xcols]
                        y = df[ycol]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
                        
                    struct = len(Xcols)
                    
                    X_train = X_train.to_numpy().astype('float32')
                    y_train = y_train.to_numpy().astype('float32')
                    X_test = X_test.to_numpy().astype('float32')
                    y_test = y_test.to_numpy().astype('float32')
                    # X_train = np.random.rand(100, 10).astype('float32')
                    # y_train = np.random.rand(100, 1).astype('float32')
                    # X_test = np.random.rand(50, 10).astype('float32')
                    # y_test = np.random.rand(50, 1).astype('float32')
                    
                    class RegressionModel(nn.Module):
                        def __init__(self, struct):
                            super().__init__()
                            self.fc1 = nn.Linear(struct, 64)
                            self.fc2 = nn.Linear(64, 32)
                            self.fc3 = nn.Linear(32, 1)

                        def forward(self, x):
                            x = torch.relu(self.fc1(x))
                            x = torch.relu(self.fc2(x))
                            x = self.fc3(x)
                            return x

                    model = RegressionModel(struct = struct)

                    # Define loss function and optimizer
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters())

                    # Train model
                    num_epochs = 10
                    batch_size = 32
                    num_batches = X_train.shape[0] // batch_size

                    for epoch in range(num_epochs):
                        for i in range(num_batches):
                            # Get batch of data
                            start_idx = i * batch_size
                            end_idx = (i + 1) * batch_size
                            X_batch = torch.from_numpy(X_train[start_idx:end_idx])
                            y_batch = torch.from_numpy(y_train[start_idx:end_idx])

                            # Forward pass
                            y_pred = model(X_batch)

                            # Compute loss
                            loss = criterion(y_pred, y_batch)

                            # Backward pass and optimize
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        st.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

                    # Evaluate model
                    with torch.no_grad():
                        y_pred = model(torch.from_numpy(X_test))
                        test_loss = criterion(y_pred, torch.from_numpy(y_test))
                        st.write(f'Test loss: {test_loss.item():.4f}')
                        
                    # Make predictions
                    y_pred = model(torch.from_numpy(X_test)).detach().numpy()

                    from PIL import Image
                    #opening the image
                    image = Image.open('nn.png')
                    
                    #displaying the image on streamlit app
                    st.image(image, caption='Neural Network model architecture')
    with tab5:
        st.header("NLP")
        text = '''
        Analyse news articles and look at the sentiment scores! 

        https://omarzahran.medium.com/taking-a-moment-to-appreciate-the-greatness-of-lebron-james-164ef332cb7c
        '''
        st.write(text)

        if st.button('Run the model and create plot!'):
            with st.spinner('Training Transformer Model...'):
                time.sleep(8)
            st.success('Model Trained!')
            df_senti = pd.read_csv('sentiment.csv')
            st.write(df_senti)
            st.line_chart(df_senti['plot'])
            
    # # TODO: Use Ray to do HPO!!
    # with tab6:
    #     # st.write("We use distributed HPO training here!")
    #     # from sklearn.metrics import f1_score
    #     # from ray import tune

    #     # if st.button('Run HPO for RandomForestRegressor'):
    #     #     config = {"n_estimators": 3,
    #     #               "max_depth": 5, 
    #     #               "min_samples_split": 3, 
    #     #               "min_samples_leaf": 1}
    #     #     def train_eval_rf(config):
    #     #         rf = RandomForestRegressor(n_estimators=config["n_estimators"],
    #     #                                     max_depth=config["max_depth"],
    #     #                                     min_samples_split=config["min_samples_split"],
    #     #                                     min_samples_leaf=config["min_samples_leaf"],
    #     #                                     random_state=42)
    #     #         rf.fit(X_train, y_train)
    #     #         y_pred = rf.predict(X_test)
    #     #         score = f1_score(y_test, y_pred)
    #     #         return score
    #     #     search_space = {
    #     #         "n_estimators": tune.randint(100, 1000),
    #     #         "max_depth": tune.choice([5, 10, 15, 20]),
    #     #         "min_samples_split": tune.randint(2, 10),
    #     #         "min_samples_leaf": tune.randint(1, 10),
    #     #     }
    #     #     analysis = tune.run(train_eval_rf,
    #     #                 config=search_space,
    #     #                 num_samples=10,
    #     #                 metric="f1_score",
    #     #                 mode="max",
    #     #                 verbose=1)

    #     #     df = analysis.dataframe()
    #     #     st.write(df)import pandas as pd
    #     if st.button('Run HPO for Linear Regression'):
            # from sklearn.linear_model import Ridge
            # from sklearn.model_selection import GridSearchCV
            # from sklearn.metrics import mean_squared_error, r2_score
            
            


        
