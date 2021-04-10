# coding=utf-8
import pickle, os, io
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from TCN import TCN
import torch

SEED = 0
np.random.seed(SEED)

class ScoringService(object):
    # 訓練期間終了日
    #TRAIN_END = "2017-11-30"
    TRAIN_END = "2018-12-31"
    # 評価期間開始日
    #VAL_START = "2018-01-01"
    VAL_START = "2019-02-01"
    # 評価期間終了日
    #VAL_END = "2018-12-01"
    VAL_END = "2019-12-01"
    # テスト期間開始日
    #TEST_START = "2019-01-01"
    TEST_START = "2020-01-01"
    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]

    # データをこの変数に読み込む
    dfs = None
    # モデル格納用変数
    models = None
    
    # 対象の銘柄コードをこの変数に読み込む
    codes = None

    @classmethod
    def get_inputs(cls, dataset_dir='../dataset'):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv",
            "stock_price": f"{dataset_dir}/stock_price.csv",
            "stock_fin": f"{dataset_dir}/stock_fin.csv",
            "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv",
            "stock_labels": f"{dataset_dir}/stock_labels.csv"
        }
        return inputs
    
    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
            
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "base_date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            
        return cls.dfs
    
    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list["Local Code"].values
        return cls.codes

    @classmethod
    def get_model(cls, model_path='../model'):
        """Get model method
 
        Args:
            model_path (str): Path to the trained model directory.
 
        Returns:
            bool: The return value. True for success, False otherwise.
 
        """
        cls.models = TCN(input_size=5, output_size=2, num_channels=[16, 8, 4, 2], kernel_size=2, dropout=0.5)
        weight_path = os.path.join(model_path, 'model.mdl')
        cls.models.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu') ))
        cls.models.eval()

        return True
    
    #######################################################################
    # Technical Analytics                                                 #
    #######################################################################
    @classmethod
    def calc_MADR(cls, close:pd.core.series.Series, days:int) -> np.ndarray:
        '''移動平均乖離率を計算する'''
        MA = close.rolling(days).mean()
        MADR = ((close - MA) / MA).replace([np.inf, -np.inf], 0)
        return MADR.values

    @classmethod
    def calc_MXDR(cls, high:pd.core.series.Series, days:int) -> np.ndarray:
        '''最高値乖離率を計算する'''
        MX = high.rolling(days).max()
        MXDR = ((high - MX) / MX).replace([np.inf, -np.inf], 0)
        return MXDR.values
    
    @classmethod
    def calc_MNDR(cls, min_:pd.core.series.Series, days:int) -> np.ndarray:
        '''最安値乖離率を計算する'''
        MN = min_.rolling(days).min()
        MNDR = ((min_ - MN) / MN).replace([np.inf, -np.inf], 0)
        return MNDR.values

    @classmethod
    def calc_RNDR(cls, close:int) -> int:
        '''キリ番(Round Number Divergence Rate...造語)との乖離率を計算する'''
        # 10円台, 1000円台, 10000円台ではスケールが異なる。
        # 99円までは10円を基準, 9999円までは100円を基準, 10000以上は1000円基準としてみる。
        #株価は0～93600の範囲をとりうる
        if close < 100:
            RN =int(Decimal(close).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP))
        elif close < 10000:
            RN =int(Decimal(close).quantize(Decimal('1E2'), rounding=ROUND_HALF_UP))
        else:
            RN =int(Decimal(close).quantize(Decimal('1E3'), rounding=ROUND_HALF_UP))
        # 終値がキリ番の場合はゼロなり割れない為、場合分け
        if close - RN != 0:
            RNDR = (close - RN) / RN
        else:
            RNDR = 0
        return RNDR

    @classmethod
    def calc_RSI(cls, close, day):
        '''RSIを計算する'''
        RSI = (close.diff().apply(lambda x: x if x >=0 else 0).rolling(day).sum() / close.diff().abs().rolling(day).sum()).replace([np.inf, -np.inf], 0)
        return RSI.values

    @classmethod
    def add_techniacl_data(cls, df_target: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        dfにテクニカル指標を追加
        '''
        df = df_target.copy()
        
        # 対数リターン(前日比)
        df["log_R"] = np.log1p(df["EndOfDayQuote ExchangeOfficialClose"]).diff()
        
        # リターン(変化率)
        df["return_5"] = df["EndOfDayQuote ExchangeOfficialClose"].pct_change(5)
        df["return_25"] = df["EndOfDayQuote ExchangeOfficialClose"].pct_change(25)
        df["return_75"] = df["EndOfDayQuote ExchangeOfficialClose"].pct_change(75)
        
        # ヒストリカルボラティリティ
        df["HV_5"] = df['log_R'].diff().rolling(5).std()
        df["HV_10"] = df['log_R'].diff().rolling(10).std()
        df["HV_25"] = df['log_R'].diff().rolling(25).std()
        df["HV_50"] = df['log_R'].diff().rolling(50).std()
        df["HV_75"] = df['log_R'].diff().rolling(75).std()
        df["HV_100"] = df['log_R'].diff().rolling(100).std()
        
        # ヒストリカルボラティリティの移動平均
        df["MA20_HV5"] = df['HV_5'].rolling(20).mean()
        df["MA20_HV10"] = df['HV_10'].rolling(20).mean()
        df["MA20_HV25"] = df['HV_25'].rolling(20).mean()
        df["MA20_HV50"] = df['HV_50'].rolling(20).mean()
        df["MA20_HV75"] = df['HV_75'].rolling(20).mean()
        df["MA20_HV100"] = df['HV_100'].rolling(20).mean()
        
        # 移動平均乖離(Moving Average Divergence Rate)を求める
        df['MADR5'] =  cls.calc_MADR(df['EndOfDayQuote ExchangeOfficialClose'], 5)
        df['MADR25'] =  cls.calc_MADR(df['EndOfDayQuote ExchangeOfficialClose'], 25)
        df['MADR75'] =  cls.calc_MADR(df['EndOfDayQuote ExchangeOfficialClose'], 75)
        
        # 最高値との乖離
        df['MXDR5'] =  cls.calc_MXDR(df['EndOfDayQuote High'], 5)
        df['MXDR10'] =  cls.calc_MXDR(df['EndOfDayQuote High'], 10)
        df['MXDR20'] =  cls.calc_MXDR(df['EndOfDayQuote High'], 20)
        
        # 最高値との乖離
        df['MNDR5'] =  cls.calc_MNDR(df['EndOfDayQuote Low'], 5)
        df['MNDR10'] =  cls.calc_MNDR(df['EndOfDayQuote Low'], 10)
        df['MNDR20'] =  cls.calc_MNDR(df['EndOfDayQuote Low'], 20)
        
        # キリ番との乖離
        df['RNDR'] =  df['EndOfDayQuote ExchangeOfficialClose'].apply(cls.calc_RNDR)
        
        # RSI
        df['RSI'] = cls.calc_RSI(df["EndOfDayQuote ExchangeOfficialClose"], 14)
        
        # 値幅(高値-安値) / 終値: O-H_C
        df['H-L_C'] =  (df['EndOfDayQuote High'] - df['EndOfDayQuote Low']) / df['EndOfDayQuote ExchangeOfficialClose']
        df['MA5_H-L_C'] = df['H-L_C'].rolling(5).mean()
        df['MA10_H-L_C'] = df['H-L_C'].rolling(10).mean()
        df['MA25_H-L_C'] = df['H-L_C'].rolling(25).mean()
        df['MA50_H-L_C'] = df['H-L_C'].rolling(50).mean()
        df['MA75_H-L_C'] = df['H-L_C'].rolling(75).mean()
        df['MA100_H-L_C'] = df['H-L_C'].rolling(100).mean()
        
        # 欠損値は削除
        #df.dropna(inplace=True)

        # 欠損値は0とする
        # テストデータの予測で、欠損値を除外とするとエラーとなる。0とるのは古いデータのみであるため、基本的には影響なし。
        df.fillna(0, inplace=True)
        
        return df
    
    #######################################################################
    # Fundamenta; Analytics                                               #
    #######################################################################
    @classmethod
    def clean_base_date_index(cls, df_target: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        更新対応処理:修正開示が20営業日以内の場合は修正前のデータはdfから削除する。
        20日以上の場合は、修正後のデータを削除する。今回は簡単のため、営業日基準とはしない。
        '''
        # 前処理
        df = df_target.copy()
        df['Result_FinancialStatement ModifyDate'] = pd.to_datetime(df['Result_FinancialStatement ModifyDate'])

        # 修正となったインデックス取得
        modify_index = np.where(df['Result_FinancialStatement ModifyDate'] != df.index)

        # 更新日
        modify_dates = df.index[modify_index]

        # 修正元の情報開示日
        base_dates = df.loc[modify_dates]['Result_FinancialStatement ModifyDate'].values

        # 差分を取り、判定
        diff_days = modify_dates - base_dates
        mask1 = [d.days <= 20 for d in diff_days]
        mask2 = [d.days > 20 for d in diff_days]
        
        # 修正前の情報開示日が20日以内のインデックス削除
        df = df.drop(base_dates[mask1]).copy()
        
        # 更新日が20日より後のインデックス削除
        df = df.drop(modify_dates[mask2]).copy()
        
        return df

    @classmethod
    def add_growth(cls, df_target: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''前期同期比の成長率を計算する'''
        df = df_target.sort_values(['Result_FinancialStatement ReportType', 'base_date']).copy()
        
        # 売上高成長率, 営業利益成長率, 経常利益成長率, 営業利益成長率
        df['NetSales_Growth'] = df['Result_FinancialStatement NetSales'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        df['OperatingIncome_Growth'] = df['Result_FinancialStatement OperatingIncome'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        df['OrdinaryIncome_Growth'] = df['Result_FinancialStatement OrdinaryIncome'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        df['NetIncome_Growth'] = df['Result_FinancialStatement NetIncome'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

        # ReportTypeの変わり目(各レポートの最初のデータ)は0とする
        report_change_mask = df['Result_FinancialStatement ReportType'].ne(df['Result_FinancialStatement ReportType'].shift()).values
        report_change_ind = df.index[np.where(report_change_mask)]
        df.loc[report_change_ind, ['NetSales_Growth', 'OperatingIncome_Growth', 'OrdinaryIncome_Growth', 'NetIncome_Growth']] = 0
        
        # 順序を戻す(しなくてもいが)
        df = df.sort_values('base_date').copy()
        
        return df
        
    @classmethod
    def add_fundamental_data(cls, df_target: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        df = df_target.copy()
        
        # 更新対応処理
        #df = cls.clean_base_date_index(df)
        
        # 売上高営業利益率, 売上高経常利益率, 売上高当期純利益
        df['OperatingIncome_NetSales'] = (df['Result_FinancialStatement OperatingIncome'] / df['Result_FinancialStatement NetSales']).replace([np.inf, -np.inf], 0)
        df['OrdinaryIncome_NetSales'] = (df['Result_FinancialStatement OrdinaryIncome'] / df['Result_FinancialStatement NetSales']).replace([np.inf, -np.inf], 0)
        df['NetIncome_NetSales'] = (df['Result_FinancialStatement NetIncome'] / df['Result_FinancialStatement NetSales']).replace([np.inf, -np.inf], 0)
        
        # 前年度期比の売上高成長率, 営業利益成長率, 経常利益成長率, 営業利益成長率
        df = cls.add_growth(df)
        
        # 来期予想成長率
        df['Forecast_NetSales_Growth'] = (df['Forecast_FinancialStatement NetSales'] / df['Result_FinancialStatement NetSales']-1).replace([np.inf, -np.inf], 0)
        df['Forecast_OperatingIncome_Growth'] = (df['Forecast_FinancialStatement OperatingIncome'] / df['Result_FinancialStatement OperatingIncome']-1).replace([np.inf, -np.inf], 0)
        df['Forecast_OrdinaryIncome_Growth'] = (df['Forecast_FinancialStatement OrdinaryIncome'] / df['Result_FinancialStatement OrdinaryIncome']-1).replace([np.inf, -np.inf], 0)
        df['Forecast_NetIncome_Growth'] = (df['Forecast_FinancialStatement NetIncome'] / df['Result_FinancialStatement NetIncome']-1).replace([np.inf, -np.inf], 0)
        
        # 自己資本比率, ROE, ROA
        df['Capital_Ratio'] = (df['Result_FinancialStatement NetAssets'] / df['Result_FinancialStatement TotalAssets']).replace([np.inf, -np.inf], 0)
        df['ROE'] = (df['Result_FinancialStatement NetIncome'] / df['Result_FinancialStatement NetAssets']).replace([np.inf, -np.inf], 0)
        df['ROA'] = (df['Result_FinancialStatement NetIncome'] / df['Result_FinancialStatement TotalAssets']).replace([np.inf, -np.inf], 0)
        
        # キャッシュフローの正負(1, 0, -1):pn(positive, negative)
        df['CF_Operating_pn'] = np.sign(df['Result_FinancialStatement CashFlowsFromOperatingActivities']).fillna(0)
        df['CF_Financing_pn'] = np.sign(df['Result_FinancialStatement CashFlowsFromFinancingActivities']).fillna(0)
        df['CF_Investing_pn'] = np.sign(df['Result_FinancialStatement CashFlowsFromInvestingActivities']).fillna(0)

        return df

    @classmethod
    def get_past_data(cls, data_cs, one_code_tech, n=5):
        '''銘柄(code)の5日前～基準日のデータを取り出す'''
        # 取り出し対象列
        extract_cols = ['EndOfDayQuote Volume', 'log_R', 'return_5', 'return_25', 'return_75', 
                        'HV_5', 'HV_10', 'HV_25', "HV_50", 'HV_75', 'HV_100',
                        'MA20_HV5', 'MA20_HV10', 'MA20_HV25', 'MA20_HV50', 'MA20_HV75', 'MA20_HV100', 
                        'MADR5', 'MADR25', 'MADR75', 'MXDR5', 'MXDR10', 'MXDR20', 'MNDR5', 'MNDR10', 'MNDR20', 'RNDR', 
                        'RSI', 'H-L_C', 'MA25_H-L_C']

        ts_datas = []
        for base_date in data_cs.index:
            ts_num = len(one_code_tech.loc[:base_date])
            if ts_num >= 5:
                ts_data = one_code_tech.loc[:base_date][extract_cols].tail(5).values
                #print('over 5')
            else:
                #print(code, ' : under 5')
                head = np.zeros((5-ts_num, len(extract_cols)))
                if ts_num == 0:
                    ts_data = head
                else:
                    tail = one_code_tech.loc[:base_date][extract_cols].values
                    ts_data = np.concatenate([head, tail])

            ts_datas.append(ts_data)

        return np.concatenate([ts_datas])

    @classmethod
    def get_features_for_predict(cls, dfs, code, start_dt=TEST_START):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
            start_dt (str): specify date range
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        # stock_finデータを読み込み
        stock_fin = dfs["stock_fin"]

        # 1銘柄に関する財務諸表データ
        df_one_code_fund = stock_fin.loc[stock_fin['Local Code'] == code].copy()
    
        # ファンダメンタル指標を追加
        df_one_code_fund = cls.add_fundamental_data(df_one_code_fund).copy()

        # 決算日の株価を取得する(株価のない財務データは除外)
        # stock_priceを読み込み
        stock_price = dfs["stock_price"]

        # 1銘柄に関する価格情報を取り出す
        df_one_code = stock_price.loc[stock_price['Local Code'] == code].copy()
        
        df_one_code_price = df_one_code[["EndOfDayQuote Date", "EndOfDayQuote ExchangeOfficialClose"]].copy()
        df_one_code_price.rename(columns={'EndOfDayQuote Date':'base_date'}, inplace=True)
        df_one_code_fund = pd.merge(df_one_code_fund, df_one_code_price, on='base_date').copy() # このmergeでdatetimeが消える
        
        # 配当利回りを計算
        df_one_code_fund['Dividend_Yeild'] = (df_one_code_fund['Result_Dividend QuarterlyDividendPerShare'] / df_one_code_fund["EndOfDayQuote ExchangeOfficialClose"]).replace([np.inf, -np.inf], 0)
        
        # 業種区分
        stock_list = dfs["stock_list"]
        df_one_code_fund['17_Sector'] = stock_list[stock_list['Local Code'] == code]['17 Sector(Code)'].values[0]

        df_one_code_fund.set_index('base_date', inplace=True)

        return df_one_code_fund[start_dt:]

    @classmethod
    def get_model_inputs(cls, dfs, code, start_dt=TEST_START):
        # all_code_fundを読み込み
        all_code_fund = dfs["all_code_fund"]

        # 1銘柄に関するファンダメンタル情報を取り出す
        data_cs = all_code_fund.loc[all_code_fund['Local Code'] == code].copy()
        
        # stock_priceを読み込み
        stock_price = dfs["stock_price"]

        # 1銘柄に関する価格情報を取り出す
        df_one_code = stock_price.loc[stock_price['Local Code'] == code].copy()
    
        # テクニカル指標を追加
        one_code_tech = cls.add_techniacl_data(df_one_code).copy()
        data_ts = cls.get_past_data(data_cs[start_dt:], one_code_tech)

        # Torch.Tensor
        inputs_ts = torch.Tensor(data_ts) # (N, 5, 30)
        inputs_cs = torch.Tensor(data_cs[start_dt:].iloc[:, 1:].values) # base_date, Local Codeを除く # (N, 46)

        return inputs_ts, inputs_cs, data_cs[start_dt:] # stard_dtは無くてもよい

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, start_dt=TEST_START):
        """Predict method
 
        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """
        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        results = []

        # 特徴量(ファンダメンタル)を作成
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code, start_dt))
        feats_cs = pd.concat(buff)
        
        # ダミー変数化 & 不要な列を削除
        category_cols = ['CF_Operating_pn', 'CF_Financing_pn', 'CF_Investing_pn', '17_Sector', 'Result_FinancialStatement ReportType']
        feats_cs = pd.get_dummies(feats_cs, columns=category_cols).copy()
        
        delete_cols = feats_cs.iloc[:, 1:42].columns
        feats_cs.fillna(0, inplace=True)
        cls.dfs['all_code_fund'] = feats_cs.drop(delete_cols, axis=1).copy()
        
        # モデルのインプットデータ作成
        for code in codes:
            inputs_ts, inputs_cs, data_cs = cls.get_model_inputs(cls.dfs, code)
            #print(inputs_ts.size(), inputs_cs.size())
            # 予測
            predicts = cls.models(inputs_ts, inputs_cs)
            
            # 結果を以下のcsv形式で出力する
            # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
            # ２列目:label_high_20 終値→最高値への変化率
            # ３列目:label_low_20 終値→最安値への変化率
            # headerはなし、B列C列はfloat64
            # 結果格納用dfを作成
            df_result = data_cs[['Local Code']].copy()
            df_result.index = pd.to_datetime(df_result.index)
            df_result['Local Code'] = df_result.index.strftime("%Y-%m-%d-") + df_result.loc[:, "Local Code"].astype(str)

            # 予測した値を格納('label_high_20', 'label_low_20')
            df_result['label_high_20'] = predicts[:, 0].detach().numpy().copy() 
            df_result['label_low_20'] = predicts[:, 1].detach().numpy().copy()
            
            # 結果をまとめる
            results.append(df_result)
                    
        df = pd.concat(results)
        df.fillna(0, inplace=True) # 2020-11-09-7337がnullになっているため。
        df.to_csv('./result.csv')

        # 出力対象列に追加
        output_columns = ['Local Code'] # 出力対象列を定義
        output_columns.append('label_high_20')
        output_columns.append('label_low_20')

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()