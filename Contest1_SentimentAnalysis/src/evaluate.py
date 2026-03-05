import pandas as pd
import sys, os
import argparse
import time
from datetime import datetime
pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_columns', 10) # for google colab
pd.set_option('display.max_rows', 10) # for google colab

class EvaluateModel():
    def __init__(self, train_file, pred_file):
        self.train_file = train_file
        self.pred_file = pred_file

    def check_files(self):
        if not os.path.exists(self.train_file):
            print(f'NOT FOUND: {self.train_file}'); sys.exit()
        if not os.path.exists(f'{self.pred_file}'):
            print(f'NOT FOUND: {self.pred_file}'); sys.exit()
        try:
            self.train = pd.read_csv(f'{self.train_file}')
        except:
            print(f'CANNOT READ: {self.train_file}'); sys.exit()
        try:
            self.pred = pd.read_csv(f'{self.pred_file}')
        except:
            print(f'CANNOT READ: {self.pred_file}'); sys.exit()
        if not {'id','aspectCategory','polarity'} <= set(self.pred.columns):
            print(f'INCORRECT COLUMN NAME : must have "id", "aspectCategory", "polarity"'); sys.exit()

    def make_tuple_set(self):
        ### MAKE ID LIST ###
        ID_used_in_prediction = set(self.pred['id'].tolist())

        ### MAKE GOLD TUPLE SET ###
        self.gold_tuple_aspect = {(ID, aspect) for ID, aspect 
            in zip(self.train['id'], self.train['aspectCategory'])
            if ID in ID_used_in_prediction}

        self.gold_tuple_sentiment = {(ID, sentiment) for ID, sentiment 
            in zip(self.train['id'], self.train['polarity'])
            if ID in ID_used_in_prediction}

        self.gold_tuple_overall = {(ID, aspect, sentiment) for ID, aspect, sentiment 
            in zip(self.train['id'], self.train['aspectCategory'], self.train['polarity'])
            if ID in ID_used_in_prediction}

        ### MAKE PREDICTION TUPLE SET ###
        self.pred_tuple_aspect = {(ID, aspect) for ID, aspect 
            in zip(self.pred['id'], self.pred['aspectCategory'])}

        self.pred_tuple_sentiment = {(ID, sentiment) for ID, sentiment 
            in zip(self.pred['id'], self.pred['polarity'])}

        self.pred_tuple_overall = {(ID, aspect, sentiment) for ID, aspect, sentiment 
            in zip(self.pred['id'], self.pred['aspectCategory'], self.pred['polarity'])}

    def macro_PRF(self, target, classname):
        """
        PRF for each class
        target : aspect or sentiment
        classname : food, service,...
        """
        if target == 'sentiment':
            gold, pred = self.gold_tuple_sentiment, self.pred_tuple_sentiment
        elif target == 'aspect':
            gold, pred = self.gold_tuple_aspect, self.pred_tuple_aspect

        intersection = [x for x in (gold & pred) if x[1] == classname]
        support = len([x for x in gold if x[1] == classname])
        try:
            precision = len(intersection) / len([x for x in pred if x[1] == classname])
        except ZeroDivisionError:
            precision = 0
        try:
            recall = len(intersection) / len([x for x in gold if x[1] == classname])
        except ZeroDivisionError:
            recall = 0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        return [precision, recall, f1, support]

    def micro_PRF(self, target):
        if target == 'sentiment':
            gold, pred = self.gold_tuple_sentiment, self.pred_tuple_sentiment
        elif target == 'aspect':
            gold, pred = self.gold_tuple_aspect, self.pred_tuple_aspect
        elif target == 'overall':
            gold, pred = self.gold_tuple_overall, self.pred_tuple_overall

        intersection = gold & pred
        try:
            precision = len(intersection) / len(pred)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = len(intersection) / len(gold)
        except ZeroDivisionError:
            recall = 0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        return precision, recall, f1

    def report_aspect(self): # food, price, service, ambience, anecdotes
        print('=== CLASSIFICATION : ASPECT ===')
        
        # header & PRF score
        report_df = pd.DataFrame(columns=['class name','precision','recall','F1-score','support'])
        report_df.loc[0] = ['food'] + self.macro_PRF('aspect', 'food')
        report_df.loc[1] = ['price'] + self.macro_PRF('aspect', 'price')
        report_df.loc[2] = ['service'] + self.macro_PRF('aspect', 'service')
        report_df.loc[3] = ['ambience'] + self.macro_PRF('aspect', 'ambience')
        report_df.loc[4] = ['anecdotes/miscellaneous'] + self.macro_PRF('aspect', 'anecdotes/miscellaneous')
        
        # macro avg
        support = report_df['support'].sum()
        report_df.loc[5] = ['MACRO AVG', report_df['precision'].mean(), report_df['recall'].mean(), report_df['F1-score'].mean(), support]
        
        # micro avg
        pre_micro, rec_micro, f1_micro = self.micro_PRF('aspect')
        report_df.loc[6] = ['MICRO AVG', pre_micro, rec_micro, f1_micro, support]
        
        print(report_df, '\n')

    def report_sentiment(self): # positive, negative, neutral, conflict
        print('=== CLASSIFICATION : SENTIMENT ===')

        # header & PRF score
        report_df = pd.DataFrame(columns=['class name','precision','recall','F1-score','support'])
        report_df.loc[0] = ['positive'] + self.macro_PRF('sentiment', 'positive')
        report_df.loc[1] = ['negative'] + self.macro_PRF('sentiment', 'negative')
        report_df.loc[2] = ['neutral'] + self.macro_PRF('sentiment', 'neutral')
        report_df.loc[3] = ['conflict'] + self.macro_PRF('sentiment', 'conflict')
        
        # macro avg
        support = report_df['support'].sum()
        report_df.loc[4] = ['MACRO AVG', report_df['precision'].mean(), report_df['recall'].mean(), report_df['F1-score'].mean(), support]
        
        # micro avg
        pre_micro, rec_micro, f1_micro = self.micro_PRF('sentiment')
        report_df.loc[5] = ['MICRO AVG', pre_micro, rec_micro, f1_micro, support]

        print(report_df, '\n')

    def report_overall(self): #
        print('=== CLASSIFICATION : OVERALL ===')

        # header & PRF score
        report_df = pd.DataFrame(columns=['','precision','recall','F1-score','support'])
        
        # micro avg
        pre_micro, rec_micro, f1_micro = self.micro_PRF('overall')
        report_df.loc[0] = ['MICRO AVG', pre_micro, rec_micro, f1_micro, len(self.gold_tuple_overall)]

        print(report_df, '\n')

    def log_results(self, model_name, params, split, csv_file='evaluation_results.csv'):
        # Get metrics
        aspect_prec, aspect_rec, aspect_f1 = self.micro_PRF('aspect')
        sent_prec, sent_rec, sent_f1 = self.micro_PRF('sentiment')
        overall_prec, overall_rec, overall_f1 = self.micro_PRF('overall')
        
        # Create result row
        result = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Model': model_name,
            'Parameters': params,
            'Split': split,
            'Aspect_F1': aspect_f1,
            'Sentiment_F1': sent_f1,
            'Overall_F1': overall_f1,
            'Evaluated_File': self.pred_file
        }
        
        # Append to CSV
        df = pd.DataFrame([result])
        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', header=False, index=False)
        
        print(f"Results logged to {csv_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Aspect Based Sentiment Analysis')
    parser.add_argument('gold_file', help='Path to gold standard file (CSV)')
    parser.add_argument('pred_file', help='Path to prediction file (CSV)')
    parser.add_argument('--model', default='Unknown', help='Name of the model being evaluated')
    parser.add_argument('--params', default='None', help='Hyperparameters or notes')
    parser.add_argument('--split', default='test', help='Content split (train/dev/test)')
    parser.add_argument('--csv', default=None, help='Path to CSV log file (default: evaluation_{split}_log.csv)')
    
    args = parser.parse_args()
    
    # Determined default CSV name based on model
    csv_file = args.csv
    if csv_file is None:
        safe_model_name = "".join([c if c.isalnum() else "_" for c in args.model])
        csv_file = f'evaluation_{safe_model_name}.csv'
    
    EVAL = EvaluateModel(args.gold_file, args.pred_file)
    EVAL.check_files()
    EVAL.make_tuple_set()
    EVAL.report_aspect()
    EVAL.report_sentiment()
    EVAL.report_overall()
    
    if args.model != 'Unknown':
        EVAL.log_results(args.model, args.params, args.split, csv_file)
