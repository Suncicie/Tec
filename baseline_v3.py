# -*- coding: utf-8 -*-
"""
baseline 2: ad.csv (creativeID/adID/camgaignID/advertiserID/appID/appPlatform) + lr
"""

import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import gc
import numpy as np


if __name__=='__main__':
    # load data
    data_root = "dataset/pre"
    dfInstalled = pd.read_csv("%s/user_installedapps.csv"%data_root)
    dfInstalled['count'] = 1
    instslled_count = dfInstalled[['userID','count']].groupby('userID').sum().reset_index()
    instslled_count_app = dfInstalled[['appID','count']].groupby('appID').sum().reset_index()
    instslled_count_app.columns = ['appID','appID_count']
    # print instslled_count_app
    dfTrain = pd.read_csv("%s/train.csv"%data_root)
    dfTest = pd.read_csv("%s/test.csv"%data_root)
    dfAd = pd.read_csv("%s/ad.csv"%data_root)
    dfApp_Cate = pd.read_csv("%s/app_categories.csv"%data_root)
    dfAd = pd.merge(dfAd, dfApp_Cate, on="appID", how="left")
    del dfApp_Cate
    gc.collect()
    dfPosition = pd.read_csv("%s/position.csv"%data_root)
    dfUser = pd.read_csv("%s/user.csv"%data_root)
    dfUser['age'] = dfUser['age'].map(lambda x : x / 15)
    # process data
    dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
    dfTrain = pd.merge(dfTrain, dfPosition, on="positionID")
    dfTrain = pd.merge(dfTrain, dfUser, on="userID")
    # 划分小时
    dfTrain['hour'] = dfTrain['clickTime'].map(lambda x: ((x%10000)/100)/6)
    dfTrain['day'] = dfTrain['clickTime'].map(lambda x: (x/10000))
    # dfTrain['day_click_count'] = 1
    # dfTrain = dfTrain.drop_duplicates(['userID','day'])
    # day_click_count = dfTrain[['userID','day','day_click_count']].groupby(['userID','day']).sum().reset_index()
    # print day_click_count
    # dfTrain = pd.merge(dfTrain,day_click_count,on=['userID','day'],how='left')
    # 历史记录中 用户安装的app 与 app有多少用户安装
    dfTrain = pd.merge(dfTrain,instslled_count,on='userID',how='left')
    # print dfTrain.columns
    dfTrain = pd.merge(dfTrain,instslled_count_app,on='appID',how='left')
    dfTrain['count'] = dfTrain['count'].fillna(1)
    dfTrain['appID_count'] = dfTrain['appID_count'].fillna(0)
    dfTrain['app_1_category'] = dfTrain['appCategory'].map(lambda x: x / 100)
    dfTrain['app_2_category'] = dfTrain['appCategory'].map(lambda x: x % 100)


    dfTest = pd.merge(dfTest, dfAd, on="creativeID")
    dfTest = pd.merge(dfTest, dfPosition, on="positionID")
    dfTest = pd.merge(dfTest, dfUser, on="userID")
    dfTest['hour'] = dfTest['clickTime'].map(lambda x: ((x%10000)/100)/6)
    dfTest['day'] = dfTest['clickTime'].map(lambda x: (x/10000))
    # dfTest['day_click_count'] = 1
    # dfTest = dfTest.drop_duplicates(['userID','day'])
    # day_click_count = dfTest[['userID','day','day_click_count']].groupby(['userID','day']).sum().reset_index()
    # dfTest = pd.merge(dfTest,day_click_count,on=['userID','day'],how='left')
    dfTest = pd.merge(dfTest,instslled_count,on='userID',how='left')
    dfTest = pd.merge(dfTest,instslled_count_app,on='appID',how='left')
    dfTest['count'] = dfTest['count'].fillna(1)
    dfTest['appID_count'] = dfTest['appID_count'].fillna(0)
    dfTest['app_1_category'] = dfTest['appCategory'].map(lambda x: x / 100)
    dfTest['app_2_category'] = dfTest['appCategory'].map(lambda x: x % 100)

    # camgaignID_ad = dfTrain.drop_duplicates(['advertiserID','camgaignID'])
    # camgaignID['camgaignID_count'] = 1
    # camgaignID_ad = camgaignID_ad[['advertiserID','camgaignID_count']].groupby('advertiserID').sum().reset_index()
    # 
    '''
    广告位ID(positionID) 广告曝光的具体位置，如QQ空间Feeds广告位。 
    站点集合ID(sitesetID) 多个广告位的聚合，如QQ空间 
    广告位类型(positionType) 对于某些站点，人工定义的一套广告位规格分类，如Banner广告位。 
    联网方式(connectionType) 移动设备当前使用的联网方式，取值包括2G，3G，4G，WIFI，未知 
    运营商(telecomsOperator) 移动设备当前使用的运营商，取值包括中国移动，中国联通，中国电信，未知 
    账户ID(advertiserID) 腾讯社交广告的账户结构分为四级：账户——推广计划——广告——素材，账户对应一家特定的广告主。 
    推广计划ID(campaignID) 推广计划是广告的集合，类似电脑文件夹功能。广告主可以将推广平台、预算限额、是否匀速投放等条件相同的广告放在同一个推广计划中，方便管理。  
    广告ID(adID) 腾讯社交广告管理平台中的广告是指广告主创建的广告创意(或称广告素材)及广告展示相关设置，包含广告的基本信息(广告名称，投放时间等)，广告的推广目标，投放平台，投放的广告规格，所投放的广告创意，广告的受众(即广告的定向设置)，广告出价等信息。单个推广计划下的广告数不设上限。 
    素材ID(creativeID) 展示给用户直接看到的广告内容，一条广告下可以有多组素材。 
    AppID(appID) 广告推广的目标页面链接地址，即点击后想要展示给用户的页面，此处页面特指具体的App。多个推广计划或广告可以同时推广同一个App。 
    App分类(appCategory) App开发者设定的App类目标签，类目标签有两层，使用3位数字编码，百位数表示一级类目，十位个位数表示二级类目，如“210”表示一级类目编号为2，二级类目编号为10，类目未知或者无法获取时，标记为0。 
    App平台(appPlatform) App所属操作系统平台，取值为Android，iOS，未知。同一个appID只会属于一个平台。 

    '''
    # 组合特征 
    # 组合appID 和与app有关的特征
    dfTrain['appID_positionID'] = dfTrain['appID'] + dfTrain['positionID'] * 100 
    dfTest['appID_positionID'] = dfTest['appID'] + dfTest['positionID'] * 100
    # 20170606添加特征
    dfTrain['appID_appCategory'] = dfTrain['appID'] + dfTrain['appCategory'] * 100 
    dfTest['appID_appCategory'] = dfTest['appID'] + dfTest['appCategory'] * 100

    # 广告位类型
    dfTrain['positionID_connectionType'] = dfTrain['connectionType'] + dfTrain['positionID'] * 100 
    dfTest['positionID_connectionType'] = dfTest['connectionType'] + dfTest['positionID'] * 100

    dfTrain['positionType_connectionType'] = dfTrain['connectionType'] + dfTrain['positionType'] * 100
    dfTest['positionType_connectionType'] = dfTest['connectionType'] + dfTest['positionType'] * 100

    dfTrain['positionType_positionID'] = dfTrain['positionID'] + dfTrain['positionType'] * 100
    dfTest['positionType_positionID'] = dfTest['positionID'] + dfTest['positionType'] * 100
    
    dfTrain['appCategory_appPlatform'] = dfTrain['appCategory'] + dfTrain['appPlatform'] * 100
    dfTest['appCategory_appPlatform'] = dfTest['appCategory'] + dfTest['appPlatform'] * 100

    dfTrain['appID_appPlatform'] = dfTrain['appID'] + dfTrain['appPlatform'] * 100
    dfTest['appID_appPlatform'] = dfTest['appID'] + dfTest['appPlatform'] * 100

    dfTrain['appID_sitesetID'] = dfTrain['appID'] + dfTrain['sitesetID'] * 100
    dfTest['appID_sitesetID'] = dfTest['appID'] + dfTest['sitesetID'] * 100

    dfTrain['positionID_sitesetID'] = dfTrain['positionID'] + dfTrain['sitesetID'] * 100
    dfTest['positionID_sitesetID'] = dfTest['positionID'] + dfTest['sitesetID'] * 100
   
    dfTrain = dfTrain.fillna(0)

    dff_time =  dfTrain[['userID','clickTime']].groupby('userID')['clickTime'].agg([('max',np.max),('min',np.min)]).reset_index()
    dff_time['theat'] = 1
    dff_time['diff'] = dff_time['max'] - dff_time['min'] + dff_time['theat']
    dfTrain = pd.merge(dfTrain,dff_time,on='userID',how='left')
    # print dff_time
    dfTest = dfTest.fillna(0)

    dff_time =  dfTest[['userID','clickTime']].groupby('userID')['clickTime'].agg([('max',np.max),('min',np.min)]).reset_index()
    dff_time['theat'] = 1
    dff_time['diff'] = dff_time['max'] - dff_time['min'] + dff_time['theat']
    dfTest = pd.merge(dfTest,dff_time,on='userID',how='left')

    # 统计用户出现的次数
    df = dfTrain[['userID']].append(dfTest[['userID']])
    groupby_userID = df.groupby('userID').size()
    dfTrain['userIDSum'] = dfTrain['userID'].apply(lambda x:groupby_userID[x])
    dfTest['userIDSum'] = dfTest['userID'].apply(lambda x:groupby_userID[x])
    del dfUser
    del dfPosition
    gc.collect()

    # df_history = dfTrain[dfTrain['day']<=23]
    # df_history_appId = df_history.groupby('appID')['label'].mean().reset_index()
    # df_history_appId.columns = ['appID','cov']

    # df_history_test = dfTrain[dfTrain['day']>=23]
    # df_history_appId_test = df_history_test.groupby('appID')['label'].mean().reset_index()
    # df_history_appId_test.columns = ['appID','cov']
    # # print df_history_appId
    # dfTrain = dfTrain[dfTrain['day']>=23]
    # dfTrain = pd.merge(dfTrain,df_history_appId,on='appID',how='left')
    # dfTrain = dfTrain.fillna(0)

    # dfTest = pd.merge(dfTest,df_history_appId_test,on='appID',how='left')
    # dfTest = dfTest.fillna(0)
    y_train = dfTrain["label"].values
  
    # feature engineering/encoding
    enc = OneHotEncoder()

    feats = ["creativeID", "adID", "camgaignID","appID_appCategory",
            "advertiserID", "appID", "appPlatform", "appCategory","positionID","appID_positionID",
    		"sitesetID","positionType","connectionType","telecomsOperator","gender","education","age"
            ,"marriageStatus","haveBaby","appID_appPlatform","appID_sitesetID","count","positionID_sitesetID",
            "positionType_connectionType","hour","positionType_positionID","appCategory_appPlatform",
            "app_1_category","app_2_category","userID","appID_count","diff"]
           

    for i,feat in enumerate(feats):
        x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
        x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
        if i == 0:
            X_train, X_test = x_train, x_test
        
        else:
            X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))


    # xgb_cmodel = xgb.XGBClassifier().fit(dfTrain_1.values, y_train)

    # # saving to file with proper feature names
    # xgbfir.saveXgbFI(xgb_cmodel, feature_names=dfTrain_1.feature_names, OutputXlsxFile = 'demo.xlsx')
    # X_train,X_test = sparse.hstack((X_train,dfTrain['cov'].values.reshape(-1,1))),sparse.hstack((X_test,dfTest['cov'].values.reshape(-1,1)))
    # print X_te/st
    X_train,X_test = sparse.hstack((X_train,dfTrain['userIDSum'].values.reshape(-1,1))),sparse.hstack((X_test,dfTest['userIDSum'].values.reshape(-1,1)))
    print "++++++++++++++++++++"
    # print X_train.columns
    print "++++++++++++++++++++"
    # print X_test
    # model training
    # pipeline = Pipeline([
    # 	('clr',LogisticRegression())
    # 	])

    # parameters = {
    # 	'clr__penalty': ('l1', 'l2'),
    # 	'clr__C': (0.01, 0.1, 1, 10),
    # }

    # grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=3)
    lr = LogisticRegression()


    lr.fit(X_train, y_train)
    # best_parameters = grid_search.best_estimator_.get_params()
    # for param_name in sorted(parameters.keys()):
    #     print('\t%s: %r' % (param_name, best_parameters[param_name]))
    proba_test = lr.predict_proba(X_test)[:,1]


    # submission
    df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
    df.sort_values("instanceID", inplace=True)
    df.to_csv("submission.csv", index=False)
    with zipfile.ZipFile("submission_0607_2.zip", "w") as fout:
        fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)
