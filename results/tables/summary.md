# Summary at k/n = 1/12, AWGN (simulation)

| dataset   | scheme            |   snr |    psnr |   ssim |      acc | mcs       |   sem_cos |   top1_agree |
|:----------|:------------------|------:|--------:|-------:|---------:|:----------|----------:|-------------:|
| cifar     | jscc_mse          |     0 | 20.0791 | 0.6694 |   0.4198 | nan       |  nan      |     nan      |
| cifar     | jscc_sem          |     0 | 19.6845 | 0.6768 |   0.5642 | nan       |  nan      |     nan      |
| cifar     | sep_ae_capacity   |     0 | 19.6502 | 0.6001 |   0.1528 | nan       |  nan      |     nan      |
| cifar     | sep_ae_best       |     0 | 12.6284 | 0.1319 |   0.0976 | QPSK 1/2  |  nan      |     nan      |
| cifar     | jscc_mse          |     4 | 22.9318 | 0.7872 |   0.5546 | nan       |  nan      |     nan      |
| cifar     | jscc_sem          |     4 | 22.4695 | 0.7927 |   0.717  | nan       |  nan      |     nan      |
| cifar     | sep_ae_capacity   |     4 | 19.6502 | 0.6001 |   0.1528 | nan       |  nan      |     nan      |
| cifar     | sep_ae_best       |     4 | 17.7392 | 0.5214 |   0.136  | QPSK 3/4  |  nan      |     nan      |
| cifar     | jscc_mse          |    10 | 25.9926 | 0.8735 |   0.676  | nan       |  nan      |     nan      |
| cifar     | jscc_sem          |    10 | 25.4761 | 0.8738 |   0.7822 | nan       |  nan      |     nan      |
| cifar     | sep_ae_capacity   |    10 | 24.5081 | 0.8075 |   0.2966 | nan       |  nan      |     nan      |
| cifar     | sep_ae_best       |    10 | 19.6502 | 0.6001 |   0.1528 | QPSK 3/4  |  nan      |     nan      |
| cifar     | jscc_mse          |    16 | 27.3743 | 0.9    |   0.6994 | nan       |  nan      |     nan      |
| cifar     | jscc_sem          |    16 | 26.8132 | 0.8982 |   0.7992 | nan       |  nan      |     nan      |
| cifar     | sep_ae_capacity   |    16 | 26.5294 | 0.8755 |   0.6032 | nan       |  nan      |     nan      |
| cifar     | sep_ae_best       |    16 | 24.4993 | 0.8072 |   0.2962 | 64QAM 2/3 |  nan      |     nan      |
| cifar     | jscc_mse          |    20 | 27.7222 | 0.9056 |   0.706  | nan       |  nan      |     nan      |
| cifar     | jscc_sem          |    20 | 27.1451 | 0.9032 |   0.8018 | nan       |  nan      |     nan      |
| cifar     | sep_ae_capacity   |    20 | 27.3291 | 0.8959 |   0.6738 | nan       |  nan      |     nan      |
| cifar     | sep_ae_best       |    20 | 25.7084 | 0.8504 |   0.4052 | 64QAM 3/4 |  nan      |     nan      |
| kodak     | jscc_mse          |     0 | 22.9682 | 0.5535 | nan      | nan       |    0.8143 |       0.2917 |
| kodak     | jscc_sem          |     0 | 22.7294 | 0.591  | nan      | nan       |    0.7971 |       0.2083 |
| kodak     | sep_jpeg_capacity |     0 | 26.6956 | 0.7302 | nan      | nan       |    0.9316 |       0.6667 |
| kodak     | sep_jpeg_best     |     0 | 12.8172 | 0.4424 | nan      | 16QAM 1/2 |    0.359  |       0      |
| kodak     | jscc_mse          |     4 | 25.2369 | 0.6994 | nan      | nan       |    0.8891 |       0.5    |
| kodak     | jscc_sem          |     4 | 25.0851 | 0.718  | nan      | nan       |    0.8749 |       0.5417 |
| kodak     | sep_jpeg_capacity |     4 | 29.3746 | 0.8197 | nan      | nan       |    0.9786 |       0.7917 |
| kodak     | sep_jpeg_best     |     4 | 24.8223 | 0.6866 | nan      | QPSK 1/2  |    0.9255 |       0.7083 |
| kodak     | jscc_mse          |    10 | 27.1106 | 0.8212 | nan      | nan       |    0.958  |       0.7083 |
| kodak     | jscc_sem          |    10 | 27.3448 | 0.83   | nan      | nan       |    0.9489 |       0.75   |
| kodak     | sep_jpeg_capacity |    10 | 32.2735 | 0.8902 | nan      | nan       |    0.9941 |       0.9583 |
| kodak     | sep_jpeg_best     |    10 | 29.7743 | 0.8313 | nan      | 16QAM 1/2 |    0.9824 |       0.9167 |
| kodak     | jscc_mse          |    16 | 27.7629 | 0.863  | nan      | nan       |    0.9811 |       0.9167 |
| kodak     | jscc_sem          |    16 | 28.2323 | 0.8724 | nan      | nan       |    0.9754 |       0.8333 |
| kodak     | sep_jpeg_capacity |    16 | 34.4923 | 0.9252 | nan      | nan       |    0.9979 |       1      |
| kodak     | sep_jpeg_best     |    16 | 31.5616 | 0.8759 | nan      | 16QAM 3/4 |    0.9925 |       0.9583 |
| kodak     | jscc_mse          |    20 | 27.9069 | 0.8722 | nan      | nan       |    0.9855 |       0.9583 |
| kodak     | jscc_sem          |    20 | 28.4477 | 0.8822 | nan      | nan       |    0.9818 |       0.8333 |
| kodak     | sep_jpeg_capacity |    20 | 35.7026 | 0.9396 | nan      | nan       |    0.9987 |       1      |
| kodak     | sep_jpeg_best     |    20 | 32.9281 | 0.9019 | nan      | 64QAM 2/3 |    0.9961 |       1      |
| video     | jscc_mse          |     0 | 21.9252 | 0.564  | nan      | nan       |    0.763  |       0.0111 |
| video     | jscc_sem          |     0 | 21.7698 | 0.6087 | nan      | nan       |    0.7493 |       0      |
| video     | sep_jpeg_capacity |     0 | 25.2837 | 0.7256 | nan      | nan       |    0.8437 |       0.3556 |
| video     | sep_jpeg_best     |     0 | 10.4107 | 0.3807 | nan      | 16QAM 1/2 |    0.3142 |       0      |
| video     | jscc_mse          |     4 | 24.3905 | 0.7111 | nan      | nan       |    0.8631 |       0.2    |
| video     | jscc_sem          |     4 | 24.2622 | 0.7338 | nan      | nan       |    0.8504 |       0.0333 |
| video     | sep_jpeg_capacity |     4 | 28.64   | 0.8255 | nan      | nan       |    0.9362 |       0.5222 |
| video     | sep_jpeg_best     |     4 | 23.7151 | 0.6958 | nan      | QPSK 1/2  |    0.8282 |       0.3111 |
| video     | jscc_mse          |    10 | 26.5926 | 0.8385 | nan      | nan       |    0.938  |       0.4667 |
| video     | jscc_sem          |    10 | 26.6227 | 0.8404 | nan      | nan       |    0.9317 |       0.4444 |
| video     | sep_jpeg_capacity |    10 | 31.9473 | 0.8994 | nan      | nan       |    0.9747 |       0.6889 |
| video     | sep_jpeg_best     |    10 | 29.1056 | 0.8383 | nan      | 16QAM 1/2 |    0.9421 |       0.5778 |
| video     | jscc_mse          |    16 | 27.3866 | 0.8831 | nan      | nan       |    0.9681 |       0.5222 |
| video     | jscc_sem          |    16 | 27.5097 | 0.8803 | nan      | nan       |    0.9632 |       0.5111 |
| video     | sep_jpeg_capacity |    16 | 34.1907 | 0.9337 | nan      | nan       |    0.9897 |       0.8    |
| video     | sep_jpeg_best     |    16 | 31.1772 | 0.8849 | nan      | 16QAM 3/4 |    0.9692 |       0.7111 |
| video     | jscc_mse          |    20 | 27.574  | 0.893  | nan      | nan       |    0.9761 |       0.6444 |
| video     | jscc_sem          |    20 | 27.7231 | 0.8897 | nan      | nan       |    0.9715 |       0.5    |
| video     | sep_jpeg_capacity |    20 | 35.3171 | 0.9468 | nan      | nan       |    0.9937 |       0.8556 |
| video     | sep_jpeg_best     |    20 | 32.6686 | 0.9116 | nan      | 64QAM 2/3 |    0.981  |       0.7222 |
