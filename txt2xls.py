#Purpose: txt converts to excel
#Author: Shroomi
#Created: 22/01/2017
#Updated: 22/01/2017

import datetime
import time
import os
import sys
import xlwt
def txt2xls(filename,xlsname):
    print 'converting xls ... '
    f = open(filename)
    x = 0                #the beginning location of excel y
    y = 0                #the beginning location of excel x
    xls=xlwt.Workbook()
    sheet = xls.add_sheet('sheet1',cell_overwrite_ok=True) #the method to produce excel
    while True:  #loop, read the txt file
        line = f.readline() #read the txt file line by line
        #print 'line type:', type(line)
        if not line:  #if there is nothing, quit
            break
        if line[0] == '0':
            #print line
            line = line.strip('0')
            line = line.lstrip('\t')
            for i in line.split('\t'):#write the context in the line x
                item=i.strip().decode('utf8')
                sheet.write(x,y,item)
                y += 1 #the new column
            x += 1 #the new line
            y = 0  #initialize the first column
    f.close()
    xls.save(xlsname+'.xls') #save

if __name__ == "__main__":
    filename = 'train.txt'
    xlsname  = 'neg_train'
    txt2xls(filename,xlsname)