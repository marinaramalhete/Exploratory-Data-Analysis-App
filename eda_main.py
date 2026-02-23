# Import dependencies
import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

sns.set()
px.defaults.width = 800
px.defaults.height = 500

# Functions Exploratory Analysis
class EDA:

    def __init__(self, dataframe):
        self.df = dataframe
        self.columns = self.df.columns
        self.num_vars = self.df.select_dtypes(include=[np.number]).columns
        self.cat_vars = self.df.select_dtypes(include=["object"]).columns

    def box_plot(self, main_var, col_x=None, hue=None):
        return px.box(self.df, x=col_x, y=main_var, color=hue)

    def violin(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="whitegrid")
        return sns.violinplot(x=col_x, y=main_var, hue=hue,
                    data=self.df, palette="husl", split=split)

    def swarmplot(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="whitegrid")
        return sns.swarmplot(x=col_x, y=main_var, hue=hue,
                    data=self.df, palette="husl", dodge=split)
    
    def histogram_num(self, main_var, hue=None, bins=None, ranger=None):
        return px.histogram(
            self.df[self.df[main_var].between(left=ranger[0], right=ranger[1])],
            x=main_var, nbins=bins, color=hue, marginal='violin')

    def scatter_plot(self, col_x,col_y,hue=None, size=None):
        return px.scatter(self.df, x=col_x, y=col_y, color=hue,size=size)

    def bar_plot(self, col_y, col_x, hue=None):
        return px.bar(self.df, x=col_x, y=col_y,color=hue)
        
    def line_plot(self, col_y,col_x,hue=None, group=None):
        return px.line(self.df, x=col_x, y=col_y,color=hue, line_group=group)

    def CountPlot(self, main_var, hue=None):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        chart = sns.countplot(x=main_var, data=self.df, hue=hue, palette='pastel')
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    def heatmap_vars(self,cols, func = np.mean):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        chart = sns.heatmap(self.df.pivot_table(index =cols[0], columns =cols[1],  values =cols[2], aggfunc=func, fill_value=0).dropna(axis=1), annot=True, annot_kws={"size": 7}, linewidths=.5)
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    def Corr(self, cols=None, method = 'pearson'):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        if len(cols) != 0:
            corr = self.df[cols].select_dtypes('number').corr(method=method)
        else:
            corr = self.df.select_dtypes('number').corr(method=method)
        chart = sns.heatmap(corr, annot=True, annot_kws={"size": 7}, linewidths=.5)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=30)
        chart.set_yticklabels(chart.get_yticklabels(), rotation=30)
        return chart
   
    def DistPlot(self, main_var):
        sns.set(style="whitegrid")
        fig, ax = plt.subplots()
        sns.histplot(self.df[main_var], kde=True, color='c', ax=ax)
        sns.rugplot(self.df[main_var], color='c', ax=ax)
        return fig
   
@st.cache_data
def get_data(file):
    df = pd.read_csv(file)
    return df

@st.cache_data
def get_stats(_df):
    stats_num = _df.describe()
    if _df.select_dtypes("object").empty:
        return stats_num.transpose(), None
    if _df.select_dtypes(np.number).empty:
        return None, _df.describe(include="object").transpose()
    else:
        return stats_num.transpose(), _df.describe(include="object").transpose()

@st.cache_data
def get_info(_df):
    return pd.DataFrame({'types': _df.dtypes, 'nan': _df.isna().sum(), 'nan%': round((_df.isna().sum()/len(_df))*100,2), 'unique':_df.nunique()})

def input_null(df, col, radio):
    df_inp = df.copy()

    if radio == 'Mean':
        st.write("Mean:", df[col].mean())
        df_inp[col] = df[col].fillna(df[col].mean())
    
    elif radio == 'Median':
        st.write("Median:", df[col].median())
        df_inp[col] = df[col].fillna(df[col].median())

    elif radio == 'Mode':
        for i in col:
            st.write(f"Mode {i}:", df[i].mode()[0])
            df_inp[i] = df[i].fillna(df[i].mode()[0])
        
    elif radio == 'Repeat last valid value':
        df_inp[col] = df[col].ffill()

    elif radio == 'Repeat next valid value':
        df_inp[col] = df[col].bfill()

    elif radio == 'Value':
        for i in col:
            number = st.number_input(f'Insert a number to fill missing values in {i}', format='%f', key=i)
            df_inp[i] = df[i].fillna(number)
    
    elif radio == 'Drop rows with missing values':
        if not isinstance(col, list):
            col = [col]
        df_inp = df.dropna(axis=0, subset=col)
        st.markdown("Rows dropped!")
        st.write('raw # of rows ', df.shape[0], ' || preproc # of rows ', df_inp.shape[0])

    st.table(get_na_info(df_inp, df, col)) 
    
    return df_inp

def input_null_cat(df, col, radio):
    df_inp = df.copy()

    if radio == 'Text':
        for i in col:
            user_text = st.text_input(f'Replace missing values in {i} with', key=i)
            df_inp[i] = df[i].fillna(user_text)
    
    elif radio == 'Drop rows with missing values':
        if not isinstance(col, list):
            col = [col]
        df_inp = df.dropna(axis=0, subset=col)
        st.markdown("Rows dropped!")
        st.write('raw # of rows ', df.shape[0], ' || preproc # of rows ', df_inp.shape[0])

    st.table(pd.concat([get_info(df[col]),get_info(df_inp[col])], axis=0))
    
    return df_inp

@st.cache_data
def get_na_info(_df_preproc, _df, col):
    raw_info = pd_of_stats(_df, col)
    prep_info = pd_of_stats(_df_preproc, col)
    return raw_info.join(prep_info, lsuffix='_raw', rsuffix='_prep').T

@st.cache_data
def pd_of_stats(_df, col):
    #Descriptive Statistics
    stats = dict()
    stats['Mean']  = _df[col].mean()
    stats['Std']   = _df[col].std()
    stats['Var'] = _df[col].var()
    stats['Kurtosis'] = _df[col].kurtosis()
    stats['Skewness'] = _df[col].skew()
    stats['Coefficient Variance'] = stats['Std'] / stats['Mean']
    return pd.DataFrame(stats, index = col).T.round(2)

@st.cache_data
def pf_of_info(_df, col):
    info = dict()
    info['Type'] =  _df[col].dtypes
    info['Unique'] = _df[col].nunique()
    info['n_zeros'] = (len(_df) - np.count_nonzero(_df[col]))
    info['p_zeros'] = round(info['n_zeros'] * 100 / len(_df),2)
    info['nan'] = _df[col].isna().sum()
    info['p_nan'] =  (_df[col].isna().sum() / _df.shape[0]) * 100
    return pd.DataFrame(info, index = col).T.round(2)

@st.cache_data
def pd_of_stats_quantile(_df, col):
    df_no_na = _df[col].dropna()
    stats_q = dict()

    stats_q['Min'] = _df[col].min()
    label = {0.25:"Q1", 0.5:'Median', 0.75:"Q3"}
    for percentile in np.array([0.25, 0.5, 0.75]):
        stats_q[label[percentile]] = df_no_na.quantile(percentile)
    stats_q['Max'] = _df[col].max()
    stats_q['Range'] = stats_q['Max']-stats_q['Min']
    stats_q['IQR'] = stats_q['Q3']-stats_q['Q1']
    return pd.DataFrame(stats_q, index = col).T.round(2)    

@st.cache_data
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

def plot_univariate(obj_plot, main_var, radio_plot_uni):
    
    if radio_plot_uni == 'Histogram' :
        st.subheader('Histogram')
        bins, range_ = None, None
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None))
        bins_ = st.sidebar.slider('Number of bins optional', value = 50)
        range_ = st.sidebar.slider(
            'Choose range optional',
            float(obj_plot.df[main_var].min()),
            float(obj_plot.df[main_var].max()),
            (float(obj_plot.df[main_var].min()), float(obj_plot.df[main_var].max())))    
        if st.sidebar.button('Plot histogram chart'):
            st.plotly_chart(obj_plot.histogram_num(main_var, hue_opt, bins_, range_))
    
    if radio_plot_uni ==('Distribution Plot'):
        st.subheader('Distribution Plot')
        if st.sidebar.button('Plot distribution'):
            fig = obj_plot.DistPlot(main_var)
            st.pyplot(fig)  

    if radio_plot_uni == 'BoxPlot' :
        st.subheader('Boxplot')
        # col_x, hue_opt = None, None
        col_x  = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None), key ='uni_boxplot_x')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key ='uni_boxplot_hue')
        if st.sidebar.button('Plot boxplot chart'):
            st.plotly_chart(obj_plot.box_plot(main_var,col_x, hue_opt))

def plot_multivariate(obj_plot, radio_plot):

    if radio_plot == ('Boxplot'):
        st.subheader('Boxplot')
        col_y  = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key ='mv_boxplot_y')
        col_x  = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None), key ='mv_boxplot_x')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key ='mv_boxplot_hue')
        if st.sidebar.button('Plot boxplot chart'):
            st.plotly_chart(obj_plot.box_plot(col_y,col_x, hue_opt))
    
    if radio_plot == ('Violin'):
        st.subheader('Violin')
        col_y  = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='mv_violin_y')
        col_x  = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None),key='mv_violin_x')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='mv_violin_hue')
        split = st.sidebar.checkbox("Split",key='mv_violin_split')
        if st.sidebar.button('Plot violin chart'):
            fig, ax = plt.subplots()
            obj_plot.violin(col_y,col_x, hue_opt, split)
            st.pyplot(fig)
    
    if radio_plot == ('Swarmplot'):
        st.subheader('Swarmplot')
        col_y = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='mv_swarm_y')
        col_x = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None),key='mv_swarm_x')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='mv_swarm_hue')
        split = st.sidebar.checkbox("Split", key ='mv_swarm_split')
        if st.sidebar.button('Plot swarmplot chart'):
            fig, ax = plt.subplots()
            obj_plot.swarmplot(col_y,col_x, hue_opt, split)
            st.pyplot(fig)

    def pretty(method):
        return method.capitalize()

    if radio_plot == ('Correlation'):
        st.subheader('Heatmap Correlation Plot')
        correlation = st.sidebar.selectbox("Choose the correlation method", ('pearson', 'kendall','spearman'), format_func=pretty)
        cols_list = st.sidebar.multiselect("Select columns",obj_plot.columns)
        st.sidebar.markdown("If None selected, it will plot the correlation of all numeric variables.")
        if st.sidebar.button('Plot heatmap chart'):
            fig, ax = plt.subplots()
            obj_plot.Corr(cols_list, correlation)
            st.pyplot(fig)

    def map_func(function):
        dic = {np.mean:'Mean', np.sum:'Sum', np.median:'Median'}
        return dic[function]
    
    if radio_plot == ('Heatmap'):
        st.subheader('Heatmap between vars')
        st.markdown(" In order to plot this chart remember that the order of the selection matters, \
            chooose in order the variables that will build the pivot table: row, column and value.")
        cols_list = st.sidebar.multiselect("Select 3 variables (2 categorical and 1 numeric)",obj_plot.columns, key= 'heatmapvars')
        agg_func = st.sidebar.selectbox("Choose one function to aggregate the data", (np.mean, np.sum, np.median), format_func=map_func)
        if st.sidebar.button('Plot heatmap between vars'):
            fig, ax = plt.subplots()
            obj_plot.heatmap_vars(cols_list, agg_func)
            st.pyplot(fig)
    
    if radio_plot == ('Histogram'):
        st.subheader('Histogram')
        col_hist = st.sidebar.selectbox("Choose main variable", obj_plot.num_vars, key = 'hist')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'hist')
        bins_, range_ = None, None
        bins_ = st.sidebar.slider('Number of bins optional', value = 30)
        range_ = st.sidebar.slider(
                'Choose range optional',
                float(obj_plot.df[col_hist].min()),
                float(obj_plot.df[col_hist].max()),
                (float(obj_plot.df[col_hist].min()), float(obj_plot.df[col_hist].max())))    
        if st.sidebar.button('Plot histogram chart'):
                st.plotly_chart(obj_plot.histogram_num(col_hist, hue_opt, bins_, range_))

    if radio_plot == ('Scatterplot'): 
        st.subheader('Scatter plot')
        col_x = st.sidebar.selectbox("Choose x variable (numerical)", obj_plot.num_vars, key = 'mv_scatter_x')
        col_y = st.sidebar.selectbox("Choose y variable (numerical)", obj_plot.num_vars, key = 'mv_scatter_y')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key = 'mv_scatter_hue')
        size_opt = st.sidebar.selectbox("Size (numerical) optional",obj_plot.columns.insert(0,None), key = 'mv_scatter_size')
        if st.sidebar.button('Plot scatter chart'):
            st.plotly_chart(obj_plot.scatter_plot(col_x,col_y, hue_opt, size_opt))

    if radio_plot == ('Countplot'):
        st.subheader('Count Plot')
        col_count_plot = st.sidebar.selectbox("Choose main variable",obj_plot.columns, key = 'mv_count_var')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'mv_count_hue')
        if st.sidebar.button('Plot Countplot'):
            fig, ax = plt.subplots()
            obj_plot.CountPlot(col_count_plot, hue_opt)
            st.pyplot(fig)
    
    if radio_plot == ('Barplot'):
        st.subheader('Barplot') 
        col_y = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='mv_bar_y')
        col_x = st.sidebar.selectbox("Choose x variable (categorical)", obj_plot.columns,key='mv_bar_x')
        hue_opt = st.sidebar.selectbox("Hue (categorical/numerical) optional", obj_plot.columns.insert(0,None),key='mv_bar_hue')
        if st.sidebar.button('Plot barplot chart'):
            st.plotly_chart(obj_plot.bar_plot(col_y,col_x, hue_opt))

    if radio_plot == ('Lineplot'):
        st.subheader('Lineplot') 
        col_y = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='mv_line_y')
        col_x = st.sidebar.selectbox("Choose x variable (categorical)", obj_plot.columns,key='mv_line_x')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='mv_line_hue')
        group = st.sidebar.selectbox("Group color (categorical) optional", obj_plot.columns.insert(0,None),key='mv_line_group')
        if st.sidebar.button('Plot lineplot chart'):
            st.plotly_chart(obj_plot.line_plot(col_y,col_x, hue_opt, group))
    
    
def main():

    st.title('Exploratory Data Analysis :mag:')
    st.header('Analyze the descriptive statistics and the distribution of your data. Preview and save your graphics.')
    
    file  = st.file_uploader('Upload your file (.csv)', type = 'csv')
 
    if file is not None:
        
        df = get_data(file)

        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=["object"]).columns

        def basic_info(df):
            st.header("Data")
            st.write('Number of observations', df.shape[0]) 
            st.write('Number of variables', df.shape[1])
            st.write('Number of missing (%)',((df.isna().sum().sum()/df.size)*100).round(2))

        #Visualize data
        basic_info(df)
        
        #Sidebar Menu
        options = ["View statistics", "Statistic univariate", "Statistic multivariate"]
        menu = st.sidebar.selectbox("Menu options", options)

        #Data statistics
        df_info = get_info(df)   
        if (menu == "View statistics"):
            df_stat_num, df_stat_obj = get_stats(df)
            st.markdown('**Numerical summary**')
            st.table(df_stat_num)
            st.markdown('**Categorical summary**')
            st.table(df_stat_obj)
            st.markdown('**Missing Values**')
            st.table(df_info)

        eda_plot = EDA(df) 

        # Visualize data

        if (menu =="Statistic univariate" ):
            st.header("Statistic univariate")
            st.markdown("Provides summary statistics of only one variable in the raw dataset.")
            main_var = st.selectbox("Choose one variable to analyze:", df.columns.insert(0,None))

            if main_var in numeric_features:
                if main_var != None:
                    st.subheader("Variable info")
                    st.table(pf_of_info(df, [main_var]).T)
                    st.subheader("Descriptive Statistics")
                    st.table((pd_of_stats(df, [main_var])).T)
                    st.subheader("Quantile Statistics") 
                    st.table((pd_of_stats_quantile(df, [main_var])).T) 
                    
                    chart_univariate = st.sidebar.radio('Chart', ('None','Histogram', 'BoxPlot', 'Distribution Plot'))
                    
                    plot_univariate(eda_plot, main_var, chart_univariate)

            if main_var in categorical_features:
                st.table(df[main_var].describe(include = "object"))
                st.bar_chart(df[main_var].value_counts().to_frame())

            st.sidebar.subheader("Explore other categorical variables!")
            var = st.sidebar.selectbox("Check its unique values and its frequency:", df.columns.insert(0,None))
            if var !=None:
                aux_chart = df[var].value_counts(dropna=False).to_frame()
                data = st.sidebar.table(aux_chart.style.bar(color='#3d66af'))

        if (menu =="Statistic multivariate" ):
            st.header("Statistic multivariate")

            st.markdown('Here you can visualize your data by choosing one of the chart options available on the sidebar!')
               
            st.sidebar.subheader('Data visualization options')
            radio_plot = st.sidebar.radio(
                'Choose plot style',
                ('Correlation', 'Boxplot', 'Violin', 'Swarmplot', 'Heatmap',
                 'Histogram', 'Scatterplot', 'Countplot', 'Barplot', 'Lineplot'))

            plot_multivariate(eda_plot, radio_plot)


        st.sidebar.title('Hi, everyone!')
        st.sidebar.info(
            'I hope this app data explorer tool is useful for you!\n'
            'You can find me here:\n'
            'www.linkedin.com/in/marinaramalhete')


if __name__ == '__main__':
    main()