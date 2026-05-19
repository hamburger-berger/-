import os
import time
import random

import baostock as bs
import pandas as pd
from tqdm import tqdm


a_share_codes = [
    "600519", "600036", "601318", "600900", "601398", "300750", "000333", "601899", "002594", "600030",
    "601088", "300760", "601628", "002714", "300498", "000876", "000895", "603477", "605296", "002100",
    "002840", "002567", "600975", "300761", "002299", "002157", "601012", "600438", "300274", "002460",
    "002466", "603799", "688223", "688599", "300014", "002202", "002129", "688303", "600905", "601919",
    "601872", "600026", "601018", "601298", "600018", "001872", "600150", "000039", "603167", "001205"
]


start_date = "2019-01-01"
end_date = "2025-12-31"

# 1 = 后复权
# 2 = 前复权
# 3 = 不复权
adjustflag = "2"

output_dir = "ashare_data_2019_2025_baostock"
os.makedirs(output_dir, exist_ok=True)

failed_codes = []
all_stocks_data = pd.DataFrame()


def to_baostock_code(code):
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh." + code
    else:
        return "sz." + code


def fetch_one_stock(code, max_retry=3):
    bs_code = to_baostock_code(code)

    fields = (
        "date,code,open,high,low,close,preclose,"
        "volume,amount,adjustflag,turn,tradestatus,pctChg,"
        "peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
    )

    for attempt in range(1, max_retry + 1):
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjustflag
            )

            if rs.error_code != "0":
                print(f"\n{code} 第 {attempt} 次失败：{rs.error_msg}")
                time.sleep(random.uniform(2, 5))
                continue

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            df = pd.DataFrame(data_list, columns=rs.fields)

            if df.empty:
                print(f"\n{code} 返回空数据，第 {attempt} 次")
                time.sleep(random.uniform(2, 5))
                continue

            df.insert(0, "股票代码", code)

            numeric_cols = [
                "open", "high", "low", "close", "preclose",
                "volume", "amount", "turn", "pctChg",
                "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"
            ]

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            print(f"\n抓取 {code} 第 {attempt} 次异常：{repr(e)}")
            time.sleep(random.uniform(3, 6))

    return pd.DataFrame()


lg = bs.login()

print("BaoStock login error_code:", lg.error_code)
print("BaoStock login error_msg:", lg.error_msg)

if lg.error_code != "0":
    raise RuntimeError("BaoStock 登录失败，请检查网络或稍后重试")


print(f"\n开始抓取 {len(a_share_codes)} 只 A 股数据，时间范围：{start_date} 到 {end_date}")

try:
    for code in tqdm(a_share_codes):
        df = fetch_one_stock(code, max_retry=3)

        if df is not None and not df.empty:
            file_path = os.path.join(output_dir, f"{code}_qfq_2019_2025.csv")
            df.to_csv(file_path, index=False, encoding="utf-8-sig")

            all_stocks_data = pd.concat([all_stocks_data, df], ignore_index=True)

            time.sleep(random.uniform(1, 3))
        else:
            failed_codes.append(code)

finally:
    bs.logout()


if not all_stocks_data.empty:
    combined_path = "all_50_stocks_2019_2025_baostock.csv"
    all_stocks_data.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"\n已生成合并总表：{combined_path}")
else:
    print("\n没有成功抓取到任何股票数据")


print("\n--- A股抓取完成 ---")

if failed_codes:
    print("以下股票抓取失败：")
    print(failed_codes)

    pd.DataFrame({"failed_code": failed_codes}).to_csv(
        "failed_codes_2019_2025_baostock.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("失败列表已保存为：failed_codes_2019_2025_baostock.csv")
else:
    print("全部股票抓取成功")
