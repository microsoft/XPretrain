Responsible AI Considerations

The proposed video-language dataset and pre-training model shows the capacity and generalization of learned VL representation which could benefit many applications of CV and NLP with a large range of uses across many domains. Each one of the uses has potential benefits and societal impacts. While we foresee that our technology could be used to find key information and improve efficiency and effectiveness for helpdesks, recommendation, retail and sales, we realize that it could also be used, in combination with new data, to fine-tune models to mislead, or otherwise harm people. We are also aware that this work uses considerable computation resources which itself, has environmental impacts. Therefore reducing the model size and computing effort is essential for future research.

Machine learning systems can display unfair behavior for different individuals or groups. This is a multi-dimensional, socio-technical challenge and is not explicitly addressed or captured in the current accuracy metrics for this research technology. In general, standardized fairness measures have not yet been agreed upon in academia or industry. We see opportunities for more work in this area to develop methods and benchmarks for measuring fairness aspects.

Given that user generated data is used, it is possible that certain demographic groups may not have enough representation. While we have balanced various video categories to mitigate for disparities, it is still likely that bias and fairness issues exist; this is an area of potential future work.  There may be a Western heteronormative bias, stereotypical depictions of historically marginalized populations and/or lack of representation among some groups. Although we have filtered the input data for explicit and violent content, it is possible that it hasn’t been totally eliminated in the training data and could have impacts on the results.

With visual generation techniques it is particularly important to do further work to prevent malicious use to misinform or harm people.

While some mitigations for potential harms can be done in the base model, it’s important to recognize that considering risks for fine tuning data for particular scenarios is critical as well. Ultimately, choosing the application scenario of any final model used in a production system will require careful consideration of potential harms specific to the scenario. 

For help or issues using the pre-trained models, please submit an issue or contact Bei Liu (bei.liu@microsoft.com) and Jianlong Fu (jianf@microsoft.com).
