from os import path, mkdir
from download_and_unzip import download_and_unzip
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import pandas as pd


class CrowdsourcedDataAgent():
    
    def __init__(self, data_path:str) -> None:
        if not path.exists(data_path):
            self._get_data()
            
        self.df_cr = self._read_data(data_path)
        self.df_cr = self._filter_malicious_answers(self.df_cr)
        self.df_cr_ans = self._aggregate_ans_majority_voting(self.df_cr)
        self.df_cr_ans = self._remove_prefixes(self.df_cr_ans)
    
    
    def _get_data(self, data_path):
        # Download crowdsourced data
        mkdir(data_path)
        download_and_unzip('https://files.ifi.uzh.ch/ddis/teaching/2023/ATAI/dataset/crowd_data/crowd_data.tsv', 
                        '../data/crowd_data/')
        download_and_unzip('https://files.ifi.uzh.ch/ddis/teaching/2023/ATAI/dataset/crowd_data/crowd_data_olat_P344FullstopCorrected.tsv', 
                        '../data/crowd_data/')
        download_and_unzip('https://files.ifi.uzh.ch/ddis/teaching/2023/ATAI/dataset/crowd_data/readme_crowd_data.txt',
                        '../data/crowd_data/')
        
        
    def _read_data(self, data_path):
        # Start by reading most recent TSV file
        return pd.read_csv(data_path + '/crowd_data_olat_P344FullstopCorrected.tsv', sep='\t')
    
    
    def _filter_malicious_answers(self, df_cr:pd.DataFrame):
        """Remove unusable answers from malicious workers

        Args:
            df_cr (pd.DataFrame): Dataframe of crowdsourced data in the format of the ATAI course
        """
        # Filter identifiable garbage answers by fast deceivers and ineligible workers 
        junk = ["I don't understand", "yes"]
        cond_junk = (df_cr['FixPosition'].isin(junk) | df_cr['FixValue'].isin(junk))
        df_cr = df_cr[~cond_junk]
        
        # Also filter out answers with an unnaturally short response time 
        cond_time = df_cr['WorkTimeInSeconds'] < 5
        df_cr = df_cr[~cond_time]
        
        return df_cr
        
        
    def _aggregate_ans_majority_voting(self, df_cr:pd.DataFrame):
        """Aggregates many differing answers per KG triple into one per triple with majority voting.

        Args:
            df_cr (pd.DataFrame): Crowdsourced data with many differing answers per KG triple

        Returns:
            pd.DataFrame: Subset of df_cr containing one answer per KG triple
        """
        # Get majority AnswerLabel
        df_cr['MajorityLabel'] = \
            df_cr.groupby(['Input1ID', 'Input2ID', 'Input3ID'])['AnswerLabel']\
            .transform(lambda x: x.mode().iat[0])
            
        # Keep only rows with majority answer label 
        df_cr_ans = df_cr.loc[df_cr['MajorityLabel'] == df_cr['AnswerLabel'], 'Input1ID':]

        # get majority answer fixes (group by inputs)
        df_cr_ans['MajorityFixPosition'] = \
            df_cr_ans.groupby(['Input1ID', 'Input2ID', 'Input3ID'])['FixPosition']\
            .transform(lambda x: x.mode(dropna=False).iat[0])
            
        df_cr_ans['MajorityFixValue'] = \
            df_cr_ans.groupby(['Input1ID', 'Input2ID', 'Input3ID'])['FixValue']\
            .transform(lambda x: x.mode(dropna=False).iat[0])
            
        df_cr_ans = \
            df_cr_ans.drop(['AnswerID', 'AnswerLabel', 'FixPosition', 'FixValue'], axis=1)\
            .drop_duplicates()
            
        return df_cr_ans
    
    
    def _remove_prefixes(self, df:pd.DataFrame):
        for col in ['Input1ID', 'Input2ID', 'Input3ID']:
            df[col] = df[col].apply(lambda x: x.split(':')[-1])
            
        return df
    
    
    def _add_inter_rater_agreement(self, df_cr:pd.DataFrame): # TODO Fix and finish
        """Compute Fleiss' Kappa inter-rater agreement per KG triple

        Args:
            df_cr (pd.DataFrame): Dataframe of crowdsourced data in the format of the ATAI course

        Returns:
            _type_: Dataframe of crowdsourced data in the format of the ATAI course with new agreement column
        """
        pivot_result = df_cr.loc[:, ['HITId', 'AnswerLabel']]\
            .pivot_table(index='HITId', columns='AnswerLabel', aggfunc='size', fill_value=0)
        pivot_result.values
        agreement = fleiss_kappa(pivot_result.values)
        df_cr['agreement'] = agreement
        
        return df_cr
    
    
    def check_and_correct_graph_ans(self, kg_triple:tuple[str, str, str]) -> tuple[bool, tuple[str, str, str]]:
        """Check and correct information from knowledge graph against crowdsourced data.
        If triple found and correct, return True; If found and incorrect, create a corrected 
        triple if fix pos and value given and return False and the corrected triple.

        Args:
            kg_triple (tuple[str, str, str]): Triple from KG

        Returns:
            tuple[bool, tuple[str, str, str]]: Correctness bool, and corrected triple if found
        """
        cond = self.df_cr_ans[['Input1ID', 'Input2ID', 'Input3ID']] == kg_triple
        cond = cond.all(axis=1)
        
        if cond.any(): # A matching triple is found in the CSD
            is_correct = self.df_cr_ans.loc[cond, 'MajorityLabel'].iat[0]
            correctness_map = {'CORRECT':True, 'INCORRECT':False}
            is_correct = correctness_map[is_correct] if is_correct else None
            
            # If triple incorrect, create a corrected triple if fix pos and value given
            fix_pos, fix_value = None, None
            fix_pos_map = {'Subject':0, 'Predicate':1, 'Object':2}
            
            if is_correct is False:
                fix_pos, fix_value = self.df_cr_ans.loc[cond, ['MajorityFixPosition', 'MajorityFixValue']].values[0]
                
                if fix_pos and fix_value:
                    fix_pos_idx = fix_pos_map[fix_pos]
                    fix_triple = list(kg_triple)
                    fix_triple[fix_pos_idx] = fix_value
                    fix_triple = tuple(fix_triple)
                    return is_correct, fix_triple
                else:
                    return is_correct, None
            
            return is_correct, None
        else:
            return None, None
        
        
    def find_crowd_ans(self, kg_tuple:tuple[str, str]) -> str:
        """Look for an answer (object) in crowd data if not found in KG graph.
        
        Cases:
        
        * An answer is found in the CSD, and it is correct: Return it.
        * An answer is found in the CSD, but it is incorrect: If the subject or predicate 
        are right and the Object is not, return the corrected Object. 
        * An answer is found in the CSD, but it is incorrect: If the object is right but 
        the subject or predicate are not, return None. 
        * No answer is found in the CSD. Return nothing.

        Args:
            kg_tuple (tuple[str, str]): Tuple of KG graph [subject, predicate]

        Returns:
            str: Object ID if found
        """
        cond = self.df_cr_ans[['Input1ID', 'Input2ID']] == kg_tuple
        cond = cond.all(axis=1)
        
        if cond.any():
            obj_id = self.df_cr_ans.loc[cond, 'Input3ID'].iat[0]
            
            is_correct = self.df_cr_ans.loc[cond, 'MajorityLabel'].iat[0]
            correctness_map = {'CORRECT':True, 'INCORRECT':False}
            is_correct = correctness_map[is_correct] if is_correct else None
            
            # Found triple is incorrect
            if is_correct is False:
                fix_pos, fix_value = self.df_cr_ans.loc[cond, ['MajorityFixPosition', 'MajorityFixValue']].values[0]

                if fix_pos == 'Object' and fix_value:
                    return fix_value
                
                # Something other than Object is incorrect
                else:
                    return None
                
            # Found triple is correct
            else:
                return obj_id