# Example of figuring out if some animal is cat or not
#-----------------------------------------------------

*Information gain*

$$\text{Information Gain} = H(p_1^\text{node})- \left(w^{\text{left}}H\left(p_1^\text{left}\right) + w^{\text{right}}H\left(p_1^\text{right}\right)\right),$$
and $H$ is the entropy, can be written as
$$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$$

log here is defined to be in base 2. 

