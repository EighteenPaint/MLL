import pandas as pd
import numpy as np
df = pd.DataFrame({
    'key1': ['a', 'a', 'b', 'b', 'a'],
    'key2': ['one', 'two', 'one', 'two', 'three'],
    'data1': np.random.randn(5),
    'data2': np.random.randn(5)
})
ass = df['data1'].groupby(df['key1'])
for name,group in ass:
    print(type(group))

# set("Hello")