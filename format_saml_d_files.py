import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f,join,sort
import sys
import os

n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()

inPath = sys.argv[1]
outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"

raw = dt.fread(inPath, columns = dt.str32)

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,Is Laundering\n"

firstTs = -1

with open(outPath, 'w') as writer:
    writer.write(header)
    for i in range(raw.nrows):
        datetime_object = datetime.strptime(raw[i,"Date"] + " " + raw[i,"Time"], '%Y-%m-%d %H:%M:%S')
        ts = datetime_object.timestamp()
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year
        hour = datetime_object.hour
        minute = datetime_object.minute # not used yet
        second = datetime_object.second # new for SAML-D, not used yet

        if firstTs == -1:
            startTime = datetime(year, month, day)
            firstTs = startTime.timestamp() - 10

        ts = ts - firstTs

        cur1 = get_dict_val(raw[i,"Received_currency"], currency)
        cur2 = get_dict_val(raw[i,"Payment_currency"], currency)

        fmt = get_dict_val(raw[i,"Payment_type"], paymentFormat)

        fromAccIdStr = raw[i,"Sender_bank_location"] + raw[i,"Sender_account"]
        fromId = get_dict_val(fromAccIdStr, account)

        toAccIdStr = raw[i,"Receiver_bank_location"] + raw[i,"Receiver_account"]
        toId = get_dict_val(toAccIdStr, account)

        # SAML-D specific: only has Amount, so we set Amount Sent = Amount Paid = Amount for now
        amountReceivedOrig = float(raw[i,"Amount"])
        amountPaidOrig = float(raw[i,"Amount"])

        isl = int(raw[i,"Is_laundering"])

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
                    (i,fromId,toId,ts,amountPaidOrig,cur2, amountReceivedOrig,cur1,fmt,isl)

        writer.write(line)

formatted = dt.fread(outPath)
formatted = formatted[:,:,sort(3)]

formatted.to_csv(outPath)
