
import pandas as pd
from Dense_test import classification_dense

qaurter_wise_prediction = pd.read_csv('./outcome/quarters.csv')

pred  = classification_dense(qaurter_wise_prediction)
pred['START_DATE'] =  pd.to_datetime(pred['START_DATE'])
pred['END_DATE'] =  pd.to_datetime(pred['END_DATE'])
pred.to_excel('C:/Users/imonb/Box/Imon Banerjee\'s Files/Koo_project/Code/outcome/quarters_dense.xlsx')
