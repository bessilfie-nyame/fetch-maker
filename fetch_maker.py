import numpy as np
import fetch_dog_data
from scipy.stats import binom_test, f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

rottweiler_tl = fetch_dog_data.get_tail_length("rottweiler")
print(np.mean(rottweiler_tl))
print(np.std(rottweiler_tl))

whippet_rescue = fetch_dog_data.get_is_rescue("whippet")
num_whippet_rescues = np.count_nonzero(whippet_rescue)
print(num_whippet_rescues)
num_whippets = np.size(whippet_rescue)
print(num_whippets)

pval = binom_test(num_whippet_rescues, n=num_whippets, p=0.08)
print(pval)

weights_whippets, weights_terriers, weights_pitbulls = (fetch_dog_data.get_weight("whippet"), fetch_dog_data.get_weight("terrier"), fetch_dog_data.get_weight("pitbull"))
_, pval_anova = f_oneway(weights_whippets, weights_terriers, weights_pitbulls)
print(pval_anova)

breed_weights = np.concatenate([weights_whippets, weights_terriers, weights_pitbulls])
labels = ['whippets'] * len(weights_whippets) + ['terriers'] * len(weights_terriers) + ['pitbulls'] * len(weights_pitbulls)
tukey_results = pairwise_tukeyhsd(breed_weights, labels, 0.05)
print(tukey_results)

poodle_colors = fetch_dog_data.get_color("poodle")
shihtzu_colors = fetch_dog_data.get_color("shihtzu")

black_count = [np.count_nonzero(poodle_colors == "black"), np.count_nonzero(shihtzu_colors == "black")]
brown_count = [np.count_nonzero(poodle_colors == "brown"), np.count_nonzero(shihtzu_colors == "brown")]
gold_count = [np.count_nonzero(poodle_colors == "gold"), np.count_nonzero(shihtzu_colors == "gold")]
gray_count = [np.count_nonzero(poodle_colors == "gray"), np.count_nonzero(shihtzu_colors == "gray")]
white_count = [np.count_nonzero(poodle_colors == "white"), np.count_nonzero(shihtzu_colors == "white")]

color_table = [black_count,
     brown_count,
     gold_count,
     gray_count,
     white_count]
color_table_df = pd.DataFrame(color_table, columns=["Poodle", "Shih Tzu"], index=["black", "brown", "gold", "gray", "white"])
print(color_table_df)

# dropping gray color count
color_table = [black_count,
     brown_count,
     gold_count,
     white_count]
chi2, pval_chi, dof, expected = chi2_contingency(color_table)
print(pval_chi)


