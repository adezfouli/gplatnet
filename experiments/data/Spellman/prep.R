names = read.csv("gene_name_label.csv")
sub_names = subset(names, std_name %in% c("ACE2", "CLN3", "FKH2", "MBP1", "MCM1", "NDD1", "SWI4", "SWI5", "SWI6"))

cdc15 = read.csv("cdc15.csv")

sub_cdc = subset(cdc15, sys_name %in% unique(sub_names$sys_name))

nrow(sub_cdc)

cycle1 = read.csv("1cycle9000.csv")

sub_cycle1 = subset(cycle1, code %in% unique(sub_names$Code))

sub_9genes = merge(x = sub_names, y = sub_cycle1, by.x = "Code", by.y = "code", all.x = FALSE)

write.csv(sub_9genes, "gene9_Spellman.csv")