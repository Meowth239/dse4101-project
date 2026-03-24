## Quick setup of the fbi library
#install.packages("devtools") # install this so you can install github repos
## Other prerequisites to install
#install.packages("stats")
#install.packages("readr")
#install.packages("pracma")
#devtools::install_github("cykbennie/fbi") # install fbi from github

library(fbi)

md = fredmd(
    file = "./data/2026-01-MD.csv",
    date_start = as.Date("1986-01-01"),
    date_end = as.Date("2026-01-01"),
    transform = TRUE
)

md <- data.frame(md)

tb3ms_data <- md[, c("date", "TB3MS")]

write.csv(md, "./data/fredmd.csv", row.names = FALSE)
write.csv(tb3ms_data, "./data/tb3ms.csv", row.names = FALSE)
