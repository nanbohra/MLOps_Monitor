import numpy as np
import pandas as pd

class ChurnDataGenerator:
    def __init__(self):

        self.distributions = {
            'non_churned' : {
                'account_age' : (300,100),
                'total_purchases' : (20,10),
                'avg_purchase_value' : (200,100),
                'days_since_purchase' : (30,20),
                'login_weekly_frequency': (20,10),
                'complaints_raised': (5,3),
                'avg_product_rating_5': (4,1),
                'email_open_rate': (0.55,0.25)
            },
            'churned' : {
                'account_age' : (180,120),
                'total_purchases' : (10,8),
                'avg_purchase_value' : (150,80),
                'days_since_purchase' :(60,40) ,
                'login_weekly_frequency': (15,10),
                'complaints_raised': (8,5),
                'avg_product_rating_5': (3,2),
                'email_open_rate': (0.25,0.15)
            }
        }

        self.features = list(self.distributions['non_churned'].keys())

    
    def generate_customer(self, id, source, base_churn_rate=0.3):
        # decide if customer churns
        will_churn = np.random.random() < base_churn_rate

        # set distribution source based on churn status
        if will_churn:
            distribution = source['churned']
        else:
            distribution = source['non_churned']

        # generate feature values for customer based on churn status / dist
        customer = {'customer_id' : id}
        for feature in distribution:
            mean, std = distribution[feature]
            value = np.random.normal(mean, std)
            
            # set some restrictions on certain feature values
            if feature == 'email_open_rate':
                value = np.clip(value, 0,1)
            elif feature == 'avg_product_rating_5':
                value= np.clip(value,0,5)
            elif feature in ['total_purchases', 'complaints_raised']:
                value = max(0, int(round(value)))
            else:
                value = max(0, value)
            
            customer[feature] = value

        customer['churned'] = 1 if will_churn else 0 

        return customer # fully formed customer profile
    

    def generate_batch(self,num_customers, base_churn_rate=0.3, use_drift=False):

        customers = []
        
        if use_drift and self.distribution_drift is not None:
            source = self.distribution_drift
        else:
            source = self.distributions

        for i in range(1,num_customers+1):
            customer = self.generate_customer(i, source, base_churn_rate)
            customers.append(customer)

        
        return pd.DataFrame(customers)
    


    def add_y_noise(self, customers, noise_rate=0.15):
        # above generation creates too clean data
        # adding noise to distributions by randomly flipping some y labels
        customers = customers.copy()

        num_flips = int(len(customers)*noise_rate)
        flip_idx = np.random.choice(len(customers), num_flips, replace=False)

        for idx in flip_idx:
            customers.loc[idx]['churned'] = 1 - customers.loc[idx]['churned']

        return customers
    
    

    def apply_feature_drift(self, magnitude=0.3):
        # magnitude scaled linearly with progress factor over batches
        # to gradually introduce feature drift

        self.distribution_drift = {
            'non_churned':{},
            'churned':{}
        }

        for churn_type in self.distribution_drift.keys():
            for feature, (mean,std) in self.distributions[churn_type].items():
                
                if feature == 'email_open_rate':
                    new_mean = mean * (1-magnitude) # decline email opening rate for both churn types
                
                elif feature == 'total_purchases':
                    new_mean = mean * (1-magnitude) # decline total purchases for both churn types
                
                elif feature == 'days_since_purchase':
                    new_mean = mean * (1+magnitude) # increase time since last purchase
                
                elif feature == 'login_weekly_frequency':
                    new_mean = mean * (1-magnitude *0.5) # reduce login freq
                
                elif feature == 'complaints_raised':
                    new_mean = mean * (1-magnitude * 0.3) # increase complaints raised

                else:
                    new_mean = mean * (1-magnitude * 0.1) # minor disruption to other feature means
            
                self.distribution_drift[churn_type][feature] = (new_mean,std)
    

def generate_drift_stream(generator, magnitude, drift_starts, ramp_period, num_batches, cust_per_batch):
    # introduce gradual feature drift starting at batch drift_starts during data generation
    streamed_data = []

    for batch_num in range(num_batches):
        if batch_num < drift_starts:
            use_magnitude = 0.0
            
        else:
            ramp_end = drift_starts + ramp_period # drift occurs over 15 batches then plateaus 
            progress = min(1.0, (batch_num - drift_starts) / (ramp_end - drift_starts))
            print(f"Progress={progress}")
            use_magnitude = magnitude * progress
        print(f"Using magnitude: {use_magnitude}")
            

        if use_magnitude > 0.0:
            generator.apply_feature_drift(use_magnitude) # update distribution gradually
            use_drift = True
        else:
            use_drift= False
            

        batch = generator.generate_batch(num_customers=cust_per_batch,use_drift=use_drift)
        batch['batch_num'] = batch_num
        batch['drift_magnitude'] = use_magnitude

        streamed_data.append(batch)

    return streamed_data


