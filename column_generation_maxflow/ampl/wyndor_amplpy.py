"""
Wyndor Glass Co. (Hillier & Lieberman) solved with amplpy.

Wyndor makes two products in three plants:
    Product 1: aluminum-framed glass doors   (profit $3,000 per batch)
    Product 2: wood-framed glass windows     (profit $5,000 per batch)

Production time (hours per batch) and weekly capacity (hours) in each plant:
              Doors   Windows   Capacity
    Plant 1     1        0         4
    Plant 2     0        2        12
    Plant 3     3        2        18

Decide how many batches of each product to make per week to maximize profit.

    max  3 x1 + 5 x2
    s.t. 1 x1          <=  4     (Plant 1)
                2 x2   <= 12     (Plant 2)
         3 x1 + 2 x2   <= 18     (Plant 3)
         x1, x2 >= 0

Setup (once):
    pip install amplpy
    python -m amplpy.modules install highs
    python -m amplpy.modules activate <your-license-uuid>   # free Community Edition
"""

from amplpy import AMPL

# ---- Data (kept in Python, passed to AMPL) ---------------------------------
products = ["Doors", "Windows"]
plants = ["Plant1", "Plant2", "Plant3"]
profit = {"Doors": 3, "Windows": 5}                       # $ thousands per batch
capacity = {"Plant1": 4, "Plant2": 12, "Plant3": 18}      # hours per week
hours = {                                                  # hours per batch
    ("Plant1", "Doors"): 1, ("Plant1", "Windows"): 0,
    ("Plant2", "Doors"): 0, ("Plant2", "Windows"): 2,
    ("Plant3", "Doors"): 3, ("Plant3", "Windows"): 2,
}

# ---- Model (AMPL syntax) ----------------------------------------------------
ampl = AMPL()
ampl.eval(r"""
    set PRODUCTS;
    set PLANTS;
    param profit{PRODUCTS};
    param capacity{PLANTS};
    param hours{PLANTS, PRODUCTS};

    var batches{PRODUCTS} >= 0;

    maximize TotalProfit: sum{p in PRODUCTS} profit[p] * batches[p];

    subject to PlantTime{k in PLANTS}:
        sum{p in PRODUCTS} hours[k, p] * batches[p] <= capacity[k];
""")

# ---- Load the data ----------------------------------------------------------
ampl.set["PRODUCTS"] = products
ampl.set["PLANTS"] = plants
ampl.param["profit"] = profit
ampl.param["capacity"] = capacity
ampl.param["hours"] = hours

# ---- Solve ------------------------------------------------------------------
ampl.option["solver"] = "highs"
ampl.solve()

# ---- Results ----------------------------------------------------------------
print(f"\nMaximum profit: ${ampl.obj['TotalProfit'].value() * 1000:,.0f} per week")
for p in products:
    print(f"  {p:8s} batches per week: {ampl.var['batches'][p].value():.2f}")

# Shadow prices (duals): extra profit ($ thousands) per additional hour of plant time
print("\nShadow prices (profit gain per extra hour of capacity):")
for k in plants:
    con = ampl.con["PlantTime"][k]
    print(f"  {k}: {con.dual():.2f}   (slack = {con.slack():.2f} hours)")
