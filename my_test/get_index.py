ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]
correlation= 3
for i in range(correlation, 0, -1):
    print(i)
    if i == correlation:
        parse_subscript_main = (
            [ALPHABET[j] for j in range(i + min(0, 1) - 1)]
            + ["ik,ekc,bci,be -> bc"]
            + [ALPHABET[j] for j in range(i + min(0, 1) - 1)]
        )
        print("========== einsum contract main %s ==========", "".join(parse_subscript_main))

    else:
        # Generate optimized contractions equations
        parse_subscript_weighting = (
            [ALPHABET[j] for j in range(i + min(0, 1))]
            + ["k,ekc,be->bc"]
            + [ALPHABET[j] for j in range(i + min(0, 1))]
        )
        parse_subscript_features = (
            ["bc"]
            + [ALPHABET[j] for j in range(i - 1 + min(0, 1))]
            + ["i,bci->bc"]
            + [ALPHABET[j] for j in range(i - 1 + min(0, 1))]
        )
        print("========== einsum contract weight %s ==========", "".join(parse_subscript_weighting))
        print("========== einsum contract features %s ==========", "".join(parse_subscript_features))
