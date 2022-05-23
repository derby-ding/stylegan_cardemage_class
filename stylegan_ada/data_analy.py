import pandas as pd
import matplotlib.pyplot as plt
'''数据分布特征，饼图'''
def distr_pie(df):
    colnum = df.shape[1]

    for i in range(0, colnum, 10):  ###最后10个不显示，防止越界
        fig, axs = plt.subplots(1, 10, sharey=False, figsize=(40, 4))
        for k in range(10):
            if i+k < colnum:
                data = df.iloc[:, i + k]
                pd.value_counts(data).plot('pie', ax=axs[k], autopct='%1.0f%%')##.set_ylabel(font)

        plt.savefig('distr_pie' + str(i) + '.png')
if __name__=='__main__':
    df = pd.DataFrame()
    df['bal'] = [117, 119, 113, 168, 144, 135, 120, 481]
    df['unb'] = [24, 119, 113, 22, 144, 135, 22, 481]