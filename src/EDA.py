from data import *


# Univariate analysis

# categorical col
def catplot(data, col):
    
    sns.catplot(kind = 'count',
               x = col,
               data = data,
               palette = 'Spectral')
    
#     plt.savefig(f'Count plot of {col}', dpi = 500)
    plt.show()

for i in cat_cols:
    catplot(data = day, col = i)


# numerical cols

def numplot(data, col):
    plt.figure(figsize = (10,5))
    plt.subplot(1, 2, 1)
    sns.distplot(data[col], color = 'salmon')
    
    plt.subplot(1, 2, 2)
    sns.boxenplot(data[col], color = 'teal')
    plt.xlabel('Density')
    plt.ylabel(f'{col}')
    
#     plt.savefig(f'Distribution plot for {col}', dpi = 500)
    plt.show()

for i in num_cols:
    numplot(day, i)


# Bivariate analysis

# num cols
#looking at relationship of num cols to that of count
g = sns.PairGrid(day[num_cols])
g.map_upper(sns.scatterplot, color = 'teal')
g.map_lower(sns.scatterplot, color = 'salmon')
g.map_diag(plt.hist)
# plt.savefig('Target vs Numerical data', dpi = 500)
plt.show()


# cat cols

for i in cat_cols:
    
    sns.catplot(data = day,
                kind = 'violin',
                x = i,
                y = 'cnt',
                palette = 'Spectral')
#     plt.savefig(f'Distribution of bike counts vs {i}', dpi = 500)
    plt.show()


# Multivariate analysis

for i in cat_cols:
    if i != 'yr':
        sns.catplot(kind = 'violin',
                    data = day,
                    x = i,
                    y = 'cnt',
                    col = 'yr',
                    palette = 'Spectral' )
#         plt.savefig('Distribution of bike counts on various {} for both years'.format(i), dpi = 500)
        plt.show()


