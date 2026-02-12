alias_map = {
    "KFMSWDKQ": "A_Rel_Air_charge",
    "KFRLSN": "A_KFRLSN",
    "TMERng_facAirChrgTq_M": "M_TMERng_facAirChrgTq",
    "IgCtl_iaOptmCrsCtrl_GM": "M_IgCtl_iaOptmCrsCtrl_GM",
    "IgCtl_iaOptmzd1_GM": "M_IgCtl_iaOptmzd1_GM",
    "KFMIOP": "M_KFMIOP",
    # ...add the rest...
}

per_map_suffix_override = {
    "KFRLSN": {"Z": "XZ"}  # A_KFRLSN_XZ instead of A_KFRLSN_Z
}

m_bytes, parsed = load_parse_build_m(
    "your_file.dcm",
    alias_map=alias_map,
    round_digits=6,
    per_map_suffix_override=per_map_suffix_override,
)

# write m_bytes to a file if needed
with open("out.m", "wb") as f:
    f.write(m_bytes)
