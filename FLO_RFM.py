# GÖREV1: VERİYİ ANLAMA VE HAZIRLAMA
import datetime as dt
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 10)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" %x)

#Adım1:
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

#Adım2:
df.head()
df.columns
df.describe().T
df.isnull().sum()
df.dtypes
df.shape
df["master_id"].nunique()
#Adım3:
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

#Adım4: isminde "date" geçen değişkenlerin veri tipini datetime64[ns]'ye çevirmek
df.dtypes
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])
# daha programatik bir çözüm
date_columns = df.loc[:, df.columns.str.contains("date")].columns
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

#Adım5:
df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total":["mean", "sum"],
                                 "customer_value_total":["mean", "sum"]})

#Adım6:
df[["master_id", "customer_value_total"]].sort_values("customer_value_total", ascending=False).head(10)
df.sort_values("customer_value_total", ascending=False)[:10]
#Adım7:
df[["master_id", "order_num_total"]].sort_values("order_num_total", ascending=False).head(10)
df.sort_values("order_num_total", ascending=False)[:10]
#Adım8:
def prepare_data(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    dataframe["first_order_date"] = pd.to_datetime(dataframe["first_order_date"])
    dataframe["last_order_date"] = pd.to_datetime(dataframe["last_order_date"])
    dataframe["last_order_date_online"] = pd.to_datetime(dataframe["last_order_date_online"])
    dataframe["last_order_date_offline"] = pd.to_datetime(dataframe["last_order_date_offline"])
    return dataframe


df = prepare_data(df)

# GÖREV2: RFM Metriklerinin Hesaplanması

#Adım1:
# Recency: Müşterinin sıcaklık, yenilik durumunu ifade eder. Analizin yapıldığı tarihten ilgili müşterinin son alışveriş
# tarihin çıkarılmasıyla elde edilen sayıdır.
# Frequency: Müşterinin sıklık durumunu ifade eder. Müşterinin toplamda yaptığı alış-veriş sayısıdır.
# Monetary: Müşteri için parasal durumu ifade eder. Bu zamana kadar yaptığı alış-verişlerde şirkete toplamda kaç para bıraktığı
#bilgisidir.

#Adım2 & Adım3:
# recency hesaplamak için analiz tarihini en son alış-veriş tarihinden iki (2) gün sonrası olarak alalım:
#en son alış-veriş tarihi:
df["last_order_date"].max()
#analiz tarihi:
today_date = dt.datetime(2021,6,1)
rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                             "order_num_total": lambda x: x,
                             "customer_value_total": lambda x: x})

#alternatif çözüm:
rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x).dt.days,
                             "order_num_total": lambda x: x,
                             "customer_value_total": lambda x: x})


#Adım4:
rfm.columns = ["recency", "frequency", "monetary"]
rfm.head()
rfm.dtypes
# Kontrol:
rfm[rfm["master_id"] == "cc294636-19f0-11eb-8d74-000d3a38a36f"]
rfm[rfm["master_id"] == "f431bd5a-ab7b-11e9-a2fc-000d3a38a36f"]
rfm[rfm["master_id"] == "d6ea1074-f1f5-11e9-9346-000d3a38a36f"]
rfm.shape
rfm.reset_index(inplace=True)
# GÖREV3: RF Skorunun Hesaplanması

#Adım1 & Adım2:
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
rfm.dtypes

#Adım3:
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

# GÖREV4: RF Skorunun Segment Olarak Tanımlanması
#Adım1 & Adım2:
seg_map = {r"[1-2][1-2]": "hibernating",
           r"[1-2][3-4]": "at_risk",
           r"[1-2]5": "cant_loose",
           r"3[1-2]": "about_to_sleep",
           r"33": "need_attention",
           r"[3-4][4-5]": "loyal_customers",
           r"41": "promising",
           r"51": "new_customers",
           r"[4-5][2-3]": "potential_loyalists",
           r"5[4-5]": "champions"}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

# GÖREV5: Aksiyon Zamanı
#Adım1:
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean","count"])
rfm.groupby("segment").agg({"recency": ["mean", "count"],
                            "frequency": ["mean", "count"],
                            "monetary": ["mean", "count"]})

#Adım2:
#a
target_segment_customer_ids = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["customer_id"]
customer_ids = df[(df["master_id"].isin(target_segment_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
customer_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
customer_ids.shape

#b
target_segment_cust_ids = rfm[rfm["segment"].isin(["cant_loose", "about_to_sleep", "new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segment_cust_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK")) |
                                        (df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]

cust_ids.to_csv("yeni_marka_hedef_müşteri2.csv", index=False)












#Adım2:
#a:
# sadık müşterileri seçelim:
rfm[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")]
# bir tane boş dataframe açalım
new_df = pd.DataFrame()
# rfm sadık müşteriler seçiminin indexlerini yeni açılan boş dataframe değişken olarak girelim:
new_df["new_master_id"] = rfm[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")].index
# new_df dataframe'ini csv'ye çevirelim
new_df.to_csv("new_master_id.csv")
# Kadın kategorisinden alışveriş yapan müşterileri seçelim
df[df["interested_in_categories_12"].str.contains("KADIN")]["master_id"]
newII_df = pd.DataFrame()
newII_df["new_customers"] = df[df["interested_in_categories_12"].str.contains("KADIN")]["master_id"]
newII_df.reset_index(inplace=True)
newII_df.drop("index", axis=1, inplace=True)
newII_df.to_csv("new_woman_customers.csv")
# Using Intersection function to find common rows
common_rows = pd.DataFrame(list(set(new_df.values).intersection(set(newII_df.values))), columns=new_df.columns)
# Using Boolean Indexing to find common rows
common_rows = new_df[new_df.isin(newII_df)].dropna()

#b:
# kaybedilmemesi gereken, uykuda olan ve yeni gelen müşterileri seçelim:
rfm[(rfm["segment"] == "cant_loose") | (rfm["segment"] == "about_to_sleep") | (rfm["segment"] == "new_customers")]
# bir tane boş dataframe açalım:
new_III_df = pd.DataFrame()
# seçilen müşterileri new_III_df'e yükleyelim:
new_III_df["new_customers"] = rfm[(rfm["segment"] == "cant_loose") | (rfm["segment"] == "about_to_sleep")
                                  | (rfm["segment"] == "new_customers")].index
new_III_df.drop("segment", axis=1, inplace=True)

# erkek ve çocuk kategorisinden alışveriş yapan müşterileri seçelim
df[df["interested_in_categories_12"].str.contains("COCUK" or "ERKEK")].head(100)
pd.set_option("display.max_rows", None)