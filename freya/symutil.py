# -*- coding: utf-8 -*-
#
# Symbolic algebra utility functions for Freya.
#
# These are basically small add-on functions for SymPy.
# This module is used by EquationSystem for symbolic expression processing.
#
# JJ 2012-08-07

# TODO/FIXME for symutil:
#
# - add a routine to take the Gateaux derivative of an expression
#   (lim t->0+  (f(x0 + t) - f(0)) / t).
# - add compatibility layer for symbols(), degree() et al. which have changed between versions

from __future__ import division

import pkg_resources  # version detection for is_new_sympy()

import sympy as sy

# Freya-specific
import util  # vercmp()


#####################################
# General utility functions
#####################################

def is_new_sympy():
    """Probe whether SymPy is version >= 0.7.0 (True), or older (False).

    The call syntax of sympy.symbols() and a few other functions
    has changed between SymPy versions.

    """
    # http://stackoverflow.com/questions/710609/checking-python-module-version-at-runtime
    sympy_ver   = pkg_resources.get_distribution("sympy").version
    minimum_ver = '0.7.0'  # minimum for new syntax

    b = (util.vercmp(sympy_ver, minimum_ver) >= 0)  # SymPy version >= 0.7.0
    return b


#####################################
# Expression simplification
#####################################

# Recursive collect. The choice of grouping (parenthesization) is of course not unique;
# this implementation aims at reducing operation count.
#
# It is geared for preprocessing SymPy expressions for NumPy evaluation. The arrays in FEM
# are typically large, so each evaluation that can be reduced away saves significant time.

def recursive_collect(expr, syms=None, **kwargs):
    """Recursively collect SymPy expression "expr" in the list of symbols "syms".

    This is a wrapper around sympy.collect() for deep rewriting of expressions.

    Breadth-first algorithm.


    If you don't know what to put in the syms kwarg, and you're aiming at reducing
    operation count, there's no need to specify it - analyze() will be automatically
    used to generate a suitable symbol list at each recursion level.

    If you want to use a custom list of syms, it is recommended to use reorder=True (default).
    If you use reorder=False, note the following:

        Consider:

        sy.collect("a*b + a*c", ["a","b","c"])
          =>  a*(b + c)     (success)

        but

        sy.collect("a*b + a*d*c", ["d","a","b","c"])
          =>  a*b + a*d*c   (failure!)

        because "d" was given before (i.e. specified as more important than) "a".
        Here collect() first extracts the part that has "d", and then tries to
        collect the remaining terms (here only "a*b") w.r.t. the remaining syms.

        But if we do:

        sy.collect("a*b + a*d*c", ["a","b","c","d"])
          =>  a*(b + d*c)     (success)

        Here all terms match "a", so the "a" is extracted and no further processing
        takes place, because no terms were left over.


        Another example:

        sy.collect("a*b*c*d + b*c*d + c*d", ["d","c","b","a"])
          => d*(a*b*c + b*c + c)
        "d" matched all terms; hence nothing was left for "c", "b" or "a".
        (This is a successful non-recursive collect.)

        sy.collect("a*b*c*d + b*c*d + c*d", ["a","d"])
          =>  a*b*c*d + d*(b*c + c)
        Now the "part left over from collecting w.r.t. 'a'" has "d" as a common factor.
        The leftover is processed next w.r.t. the next sym, "d"; hence we get a partially
        collected result.

        Contrast the previous with
        sy.collect("a*b*c*d + b*c*d + c*d", ["a","b","c","d"])
          =>  a*b*c*d + b*c*d + c*d
        Here the "part left over from collecting w.r.t. 'a'" has two terms, one of them
        matching "b". Hence, this term is extracted next, leaving unhandled only the
        last term "c*d". This remaining term is then collected w.r.t. "c".
        With this ordering, the end result is that from the user's perspective,
        nothing happens!

        sy.collect(sy.sympify("a*(b*c*d + c*d)"), "d")
          =>  a*d*(b*c + c)
        Collect always descends into Mul, even when not recursive.


    The recursive_collect algorithm works as follows:
        - Take operands of the top-level operation in expr.
        - Apply sympy.collect(operand, syms) to each non-atomic operand.
          Pass atoms through as-is.
        - Rewrite expr using the updated operands.


    If reorder=False, because sy.collect() is used internally, the same note applies
    as for sympy.collect(): syms are tried in the order they are given in the list.
    Combined with the recursive approach, this implies that:

    recursive_collect( "a*b + a*b*c + a*b*c*d", ["a", "b", "c", "d"] )

    => a * (b + b*c + b*c*d)          (after first collect())
    => a * b * (1 + c + c*d)          (after collect() applied to parenthetical expression)
    => a * b * (1 + c * (1 + d))      (after collect() applied to parenthetical expression)
    = final result, since (1 + d) cannot be simplified by collect().


    If your expression is rational, it is recommended to apply together() first:

    recursive_collect( sy.together( "a*b/e + a*b*c + a*b*c*d/e" ),
                       ["a", "b", "c", "d", "e"] )

    => a*b*(1 + c*(d + e))/e      (final result)

    Compare this with just together():

    (a*b + a*b*c*d + a*b*c*e)/e

    and collect(together()):

    a*(b + b*c*d + b*c*e)/e

    You can also apply expand(together()) before calling this function, but sometimes
    leaving out the expand() produces better results, if there is already some structure
    in the input.

    To analyze optimality, see expr.count_ops() and symutil.count_atoms().


    Parameters:
        expr = expression to collect (string, or a SymPy object (e.g. from sympify()))
        syms = list of SymPy symbols or strings. Optional.

               If specified, these symbols will be used as "syms" at each level
               of recursion. This is useful if you want to collect e.g. w.r.t. x and y only,
               while ignoring coefficients. For the ordering of the given syms, see the kwarg
               "reorder".

               If not specified, analyze() will be run at each level of recursion,
               to adaptively generate an appropriate symbol list. (Default.)

    Kwargs:
        reorder = bool. This switch affects the operation when "syms" are manually provided;
                  it has no effect when syms=None (automatic syms).

                  If True (default), the most appropriate ordering for the given syms will be
                  detected automatically for each subexpression.

                  If False, the syms are always used in the order they were given.


                  Example: collect with reordering enabled:

                  recursive_collect("d*(a*b*c + b*c + c + 1) + a*(d*c*b + c*b + b + 1)",
                                    ["b","c"], reorder=True)

                  => a*(b*(c*(d + 1) + 1) + 1) + d*(c*(b*(a + 1) + 1) + 1)

                  We see that we get optimal collection in each top-level term.
                  (Note that the first and second parts of the input have switched places
                  in the output. collect() may reorder the expressions at each level;
                  this flag does NOT control that reordering.)

                  When we do the same except without reordering of syms,

                  recursive_collect("d*(a*b*c + b*c + c + 1) + a*(d*c*b + c*b + b + 1)",
                                    ["b","c"], reorder=False)

                  => a*(b*(c*(d + 1) + 1) + 1) + d*(b*c*(a + 1) + c + 1)

                  We get an extra "c" in the second part (which was originally the first part!),
                  because "b" was specified as more important. Automatic reordering avoids this,
                  while still only collecting w.r.t. those symbols that were specified.

                  (It is difficult to craft a shorter example, because collect() even in
                   top-level mode automatically descends into the arguments of a Mul.)


    See also:
        analyze()
        sympy.collect()
        sympy.together()

        count_atoms()
        sympy.count_ops()

    """
    if type(expr) == str:
        expr = sy.sympify(expr)
    if expr.is_Atom:
        return expr

    reorder = "reorder" in kwargs  and  kwargs["reorder"]
    if syms is not None:
        autosyms = False

        # Analyze and re-order the given syms if requested. (FIXME: could be more efficient)
        if reorder:
            optimally_ordered_syms = analyze(expr)
            sympified_syms = map( lambda s: sy.sympify(s), syms )
            syms = filter( lambda obj: obj in sympified_syms, optimally_ordered_syms )
    else:
        autosyms = True
        syms = analyze(expr)

#    print "syms = %s; processing %s" % (syms, expr)

    # When reorder=True, it may happen that a subexpression does not have
    # any of the specified syms.
    #
    orig_expr = expr
    if len(syms):
        expr = sy.collect(expr, syms)

#    print "syms = %s; processed  %s  ->  %s" % (syms, orig_expr, expr)

#    print "    collected as %s" % expr
#    print "    syms = %s" % syms

    # find which args of expr are composite and which are atoms
    isatom = map( lambda obj: obj.is_Atom, expr.args )

    # rewrite the args
    newargs = []
    for j in xrange(len(isatom)):
        if isatom[j]:
            # pass atoms through as-is
            newargs.append( expr.args[j] )
        else:
            if autosyms:
                # autosyms -> reorder flag is not used
                newargs.append( recursive_collect( expr.args[j] ) )
            else:
                newargs.append( recursive_collect( expr.args[j], syms, reorder=reorder) )

#    print "    done, newargs = %s" % newargs

    # Instantiate a new object of the same type as expr (e.g. Add, Mul, ...),
    # using the updated args.
    #
    return type(expr)(*newargs)


def count_atoms(expr, **kwargs):
    """Counts the occurrences of atoms in SymPy expression "expr".

    This works like sympy.count_ops(), but counts atoms instead of operations.

    To get the unique atoms of an expr, use expr.atoms() or expr.free_symbols instead;
    this function is intended for the case where you want to get also the number of
    occurrences for each atom.

    This is useful e.g. as a collection optimality metric.


    Number atoms are ignored by default. See kwargs.

    Return value:
        depends on kwargs, below.

    Parameters:
        expr = string, SymPy expression, or iterable. In case of iterable,
               the results are collected together.

    Kwargs:
        ignore_numbers = bool, default True.

                         When True,  ignore number atoms (1, 2.0, pi, ...).
                         When False, count number atoms too.

                         Default is to ignore numbers. The atom "-1" confuses collect(),
                         and scalar numbers hardly affect the evaluation speed in FEM
                         (which is the main use case for recursive_collect()).

        visual         = bool, default False.

                         When True, return a sum expression like "2*a + 3*b + c"
                         (if expr contained 2 a's, 3 b's and one c).

                         When False, return the sum of the coefficients of the
                         visual expression (2+3+1 = 6 in the above example).

        as_dict        = bool, default False. Overrides "visual".

                         If True, return a dictionary:  atom -> #occurrences

                         If False, return either the visual or non-visual format
                         (see kwarg "visual").

    See also:
        sympy.count_ops()
        analyze()
        recursive_collect()

    """
    from sympy.core.compatibility import iterable

    visual  = "visual"  in kwargs  and  kwargs["visual"]
    as_dict = "as_dict" in kwargs  and  kwargs["as_dict"]

    # Handle one expression.
    #
    if isinstance(expr, sy.Expr):
        # Use the internal function to do the actual counting.
        #
        atomdict = _count_atoms(expr, **kwargs)

        if as_dict:
            # dictionary: name -> #occurrences
            return dict(map( lambda item: (item[0], item[1][0]), atomdict.items() ))
        elif visual:
            # sum of #occurrences * name
            counts = map( lambda item: item[1][0]*item[0], atomdict.items() )
            return reduce( lambda a,b: a+b, counts )
        else:
            # total sum only (useful e.g. as a collection optimality metric)
            counts = map( lambda item: item[1][0], atomdict.items() )  # keep just the #occurrences
            return reduce( lambda a,b: a+b, counts )

    # Handle iterables.
    # (This logic is modeled after sympy.count_ops())
    #
    elif type(expr) is dict:
        results = [count_atoms(k, **kwargs) for k in expr.iterkeys()]
        results.extend( [count_atoms(v, **kwargs) for v in expr.itervalues()] )
    elif iterable(expr):
        results = [count_atoms(i, **kwargs) for i in expr]
    elif not isinstance(expr, sy.Basic):
        results = []
    else: # it's Basic not isinstance(expr, Expr):
        assert isinstance(expr, sy.Basic)
        results = [count_atoms(a, **kwargs) for a in expr.args]

    # In case of iterables, collect the results.
    #
    # (But first check for empty results; we might have gotten a blank iterable.)
    #
    if len(results) == 0:
        if as_dict:
            return {}
        elif visual:
            return sy.S.Zero
        else:
            return 0
    ret = results[0]
    if as_dict:
        # Each result is a dict; sum corresponding values (and insert any missing keys).
        for n in xrange(1,len(results)):
            for k,v in results[n].items():
                if k in ret:
                    ret[k] += v
                else:
                    ret[k]  = v
    else:
        # Each result is either a SymPy object or an integer;
        # these types can be summed directly.
        for n in xrange(1,len(results)):
            ret += results[n]
    return ret


def _count_atoms(expr, **kwargs):
    """Counts the occurrences of atoms in SymPy expression "expr".

    Internal function; implementation for count_atoms(). Used also as-is by analyze().

    Number atoms are ignored by default. See kwargs.

    Return value:
        dictionary:  atom -> (#occurrences, topmost_level_where_seen)

        Level numbering starts from 0 at the top of the expression tree.

    Example:
        count_atoms("a + a*b + a*b*c")
    =>
        dict: { a : (3,1), b : (2,2), c : (1,2) }

        In this example, there are 3 "a"s, and they are all inside the top-level
        Add operation. The Add operation is on level 0; its arguments are on level 1.
        The 2 "b"s are inside the Mul object, which is on level 1, so the "b"s are
        on level 2. Same for the single "c"; it is inside the Mul for the a*b*c term.

        Note that in SymPy, a single Add or Mul may have any number of arguments,
        so operations of a binary nature are not necessarily binary in the tree!

    Kwargs:
        ignore_numbers = bool, default True.

                         When True,  ignore number atoms (1, 2.0, pi, ...).
                         When False, count number atoms too.

                         Default is to ignore numbers. The atom "-1" confuses collect(),
                         and scalar numbers hardly affect the evaluation speed in FEM
                         (which is the main use case for recursive_collect()).

    See also:
        analyze()
        recursive_collect()

    """
    counts = {}

    if "__level" not in kwargs:
        level = 0  # root level (top of expression tree)
    else:
        level = kwargs["__level"]

    if "ignore_numbers" in kwargs  and  kwargs["ignore_numbers"] == False:
        ignore_numbers = False
    else:
        ignore_numbers = True

    if type(expr) == str:
        expr = sy.sympify(expr)
    if expr.is_Atom:
        if not ignore_numbers  or  not expr.is_Number:
            # format: key = value, where
            #  key   = atom object
            #  value = (count, topmost level where this atom was seen)
            #
            # We start by setting the level to the current level;
            # in recursion, our caller will take care of adjusting it
            # if necessary.
            #
            counts[expr] = (1, level)
        return counts

    # descend into arguments
    for arg in expr.args:
        temp = _count_atoms(arg, __level=(level+1))
        for key,value in temp.items():
            # value = (count, topmost level where seen)
            #  ...both in subexpression "arg".
            #
            # Note that levels are counted globally, whereas the count is tallied
            # for each level of recursion separately (and hence we must sum the counts).
            #
            if key in counts:
                # Update already seen atom
                counts[key] = (counts[key][0] + value[0], min(counts[key][1], value[1]))
            else:
                # First occurrence in this subexpression, so fill it in.
                counts[key] = (value[0], value[1])

    return counts


def analyze(expr):
    """Return list of atoms in SymPy expression "expr", sorted in descending order of #occurrences in expr.

    Ties are broken by preferring atoms whose 'topmost' occurrence is nearer the top
    of the expression tree. This secondary sorting helps in some cases when there is
    e.g. one of each symbol in expr (it more likely preserves the existing parenthesization
    in such cases).

    This particular ordering is useful for expression optimization using recursive_collect()
    (default behaviour if syms=... is not given).

    Example:
        analyze("a + a*b + a*b*c")
    =>
        (a, b, c)

    See also:
        count_atoms()
        recursive_collect()
    
    """
    # NOTE on tiebreaking: sometimes sorting just by atom count does not do what is
    # expected, because atoms with the same hit count are ordered randomly.
    # Consider the following input:
    #
    # Uin*(1 + u0_x)*(u0/(__uvmag__1__))
    #
    # Ordering just by count, we would get [u0, u0_x, __uvmag__1__, Uin] (tested).
    # This causes grouping first by u0 (good), and then by u0_x... not good!
    # We should group by [Uin, u0] or [u0, Uin] before considering the other
    # symbols; these are preferred by the secondary sort criterion, as they
    # are on level 1 of the expression tree.

    atomdict = _count_atoms(expr)
    atomlist = list(atomdict.items())

    def countcmp(a, b):
        """Compare atom counts.

        Return value:
          +1 if count(a) >  count(b),
           0 if count(a) == count(b),
          -1 if count(a) <  count(b).

        """
        # item = (name, (#occurrences, min_level))
        #
        # Smaller level = nearer the top of the tree.
        #
        if a[1][0] > b[1][0]:
            return +1
        elif a[1][0] == b[1][0]:
            # break tie by secondary sort criterion
            if a[1][1] < b[1][1]:
                return +1   # prioritize smaller level
            elif a[1][1] == b[1][1]:
                return 0
            else:
                return -1
        else:
            return -1

    # Sort atoms in decreasing order of #hits.
    atomlist = reversed(sorted(atomlist, cmp=countcmp))

    # Get just the atoms, preserving the ordering.
    return map( lambda item: item[0], atomlist )


#####################################
# Taylor series expansion
#####################################

# taylor() -- Taylor series expansion compatibility routine.
#
# SymPy versions prior to 0.7.0 have a bug in series(), which prevents
# evaluation of the Taylor series at points other than x0 = 0.
#
# Old versions used to throw an exception, but especially version 0.6.7
# (in Debian Stable as of April 2012) silently computes the wrong answer.
#
# For details, see:
# http://code.google.com/p/sympy/issues/detail?id=895
# http://code.google.com/p/sympy/issues/detail?id=1334
# https://github.com/sympy/sympy/pull/61
#
# JJ 2012-04-27

def taylor(f, x, x0, order):
    """Expand f(x) as a Taylor series at x0.

    This function works around bug #895 in old SymPy versions (prior to 0.7.0).
    The bug prevented correct expansion of Taylor series for x0 != 0,
    and the correct Taylor expansion (effectively truncation) of polynomial input.
    Some versions, when trying to compute for x0 != 0, silently produce the wrong answer.

    For details, see:
    http://code.google.com/p/sympy/issues/detail?id=895
    http://code.google.com/p/sympy/issues/detail?id=1334
    https://github.com/sympy/sympy/pull/61


    Note that SymPy's O() symbol does not store the series expansion point,
    but can only properly represent O() at x0=0.

    The O(x**k) returned by this function stands for O(t**k) where t = x - x0.
    Hence, O()'s of Taylor series expanded at different points are not comparable.
    This is a limitation of the O() symbol in both 0.6.x and 0.7.x (at least up to 0.7.1).


    The running SymPy version is detected automatically.

    For SymPy 0.7.0 and above:
      Version 0.7.0 and above map the output from series() at points x0!=0.
      Because the O() symbol can only properly represent O() at x0=0,
      the x in the returned series actually means (x - x0).

      However, for our purposes this is difficult to use, so we return the result
      in a slightly different format.

      In OUR return format (to simplify things for the caller): in the O(),
      x means (x - x0), but in the rest of the expression, it means just x.

    For SymPy < 0.7.0:
      This function accounts for the expansion point by applying a change of variables;
      this makes it possible to expand at 0 (in terms of the temporary variable).
      Polynomials are subjected to further checks, and truncated if necessary.
      The O() symbol is also handled for both general input and polynomials.

    Note that our function has the same semantics regarding the O() regardless of
    SymPy version.

    """
    if is_new_sympy():  # SymPy version >= 0.7.0
        if type(x) == str:
            x = sy.sympify(x)
        if type(x0) == str:
            x0 = sy.sympify(x0)
        if type(f) == str:
            f = sy.sympify(f)

        # See docstring of sy.Expr.series for an important note on the O() term.
        #
        tay = sy.series(f, x, x0, order+1)  # semantics changed! order -> order+1

        # Change the semantics of the answer to ours.
        #
        if x0 != 0:
            for i,t in enumerate(tay.args):
                if t.is_Order:
                    order_symbol = t
                    break

            tay = tay.removeO()
            tay = tay.subs( { x : x - x0 } )
            tay += order_symbol

        return tay
    else:
        # Workaround for old SymPy versions.
        #
        # We use a change of variables to get the desired result,
        # and then do the inverse substitution in the result.
        #
        ft  = f.subs( { x : "%s + __taylor_tempvar" % x0 } )
        tay = sy.series(sy.sympify(ft), "__taylor_tempvar", 0, order)

        SymbolZero = sy.sympify("0")
        SymbolOne  = sy.sympify("1")

        # Save the O() symbol, if any. We must temporarily remove it
        # to make the backsubstitution at the end to work.
        #
        # Also, we change its variable from __taylor_tempvar to the
        # user-specified x.
        #
        # The enumerate(tay.args) and t.is_Order stuff comes from
        # the source of removeO() :)
        #
        # Polynomials don't get an O() symbol even if their order is "too high",
        # so we must work around that separately.
        #
        order_symbol = SymbolZero
        try: # ...interpreting the input as a polynomial first
            # NOTE: if we were to do this in SymPy 0.7.x, we would require
            # d = sy.Poly(f, sy.sympify("%s" % x)).degree()
            # but this code is for the old versions, so:
            d = sy.Poly(f, "%s" % x).degree

            # If successful, and the degree of the original polynomial is higher
            # than the given Taylor order, we have an asymptotic error which is
            # at most of order O(x**(order+1)).
            #
            if d > order:
                order_symbol = sy.O(sy.sympify("%s**%d" % (x, order+1)))
        except sy.PolynomialError:
            # If the input wasn't a polynomial, see if its Taylor series
            # includes an O(). If it does, grab it and change its variable
            # to x (note that our change of variables is a linear shift,
            # so this is valid).
            #
            for i,t in enumerate(tay.args):
                if t.is_Order:
                    order_symbol = t.subs( { "__taylor_tempvar" : "%s" % x } )
                    break
            # Remove the O() so that it doesn't mess up the backsubstitution.
            tay = tay.removeO()

        # Now, work around the other part of the bug:
        # for polynomial input, series() fails to discard powers
        # higher than the specified order.
        #
        # We collect the powers of our temporary variable,
        # and construct the output by including only powers
        # that are <= order.
        #
        terms = sy.collect(tay, "__taylor_tempvar", evaluate=False)
        if SymbolOne in terms:
            sol = terms[SymbolOne]  # constant term
        else:
            sol = SymbolZero        # constant term is zero
        for k in xrange(1,order+1):  # t**order is the last term to be included
            key = sy.sympify("__taylor_tempvar**%s" % k)
            if key in terms:  # some orders may be missing (think e.g. sin(x) at x = 0)
                sol += terms[key]*key  # e.g. a*x**2

        # Change variables back
        sol = sol.subs( { "__taylor_tempvar" : "%s - %s" % (x, x0) } )

        # Add back the removed O() symbol. If none, this will add 0 which is a no-op.
        sol += order_symbol
        return sol


        # DEBUG test
        #print taylor(sy.sympify("sin(x)"), "x", 1, 6)               # general example
        #print taylor(sy.sympify("sin(x)"), "x", 0, 6)               # no constant term
        #print taylor(sy.sympify("cos(x)"), "x", 0, 6)               # no first-order term
        #print taylor(sy.sympify("x**4 - a**4"), "x", "x0", 1)       # order truncation
        #print taylor(sy.sympify("1 + a*x + b*x**2"), "x", "0", 2)   # exact (no O()!)


#####################################
# Automatic linearization
#####################################

# This is a touched-up version of jjrandom2/miniprojects/misc/sympydemo.py.

def linearize(expr, var, kind, **kwargs):
    """Linearize nonlinear expression "expr" with respect to "var" around arbitrary point named "var0".

    This can be used for producing definitions of iteratives from
    e.g. polynomially nonlinear expressions.

    CAUTION:
        This is a very rudimentary implementation, which works for polynomials
        and other simple functional expressions. Expressions involving derivatives
        of var are not supported (e.g. the convection term of Navier-Stokes).
        To handle such expressions, for now, please manually calculate a Frechet
        (or Gateaux) derivative.

    Return value:
        The linearized expression (string).

    Parameters:
        expr = string or SymPy object. Original expression.

        var  = string. Variable w.r.t. which to linearize (e.g. "u").

        kind = string. Choose the linearization algorithm:
            picard = basic fixed point linearization, u -> u0.

            alpha  = extract a linear factor from a polynomially nonlinear expression.
                     This is a effectively a modified fixed point iteration.

                     Requires kwarg term=expr2, where expr2 is the desired term to extract.

                     Depending on the structure of the polynomial, this may or may not work.
                     Basically, in order to use this algorithm, you must know (symbolically)
                     one of the roots of the polynomial and then pass in term="(var - root)".

                     The name "alpha linearization" comes from
                     CSC: Numeeriset menetelmät käytännössä, p. 286.
                     (Numerical methods in practice; in Finnish).

            taylor = Taylor series expansion up to linear term.
                     Requires a good initial guess to converge; best use this
                     as the "fast" def of an iterative after its value
                     has been bootstrapped using another linearization.

    Example code:

    import sympy as sy
    import symutil

    s = sy.sympify("(u**4 - uext**4)")  # RHS of radiation BC for Poisson equation

    print "Original nonlinear expression:\\n    %s" % s
    print "Fixed point (Picard) linearization:\\n    %s" % symutil.linearize(s, "u", "picard")
    print "Alpha linearization 1:\\n    %s" % symutil.linearize(s, "u", "alpha", term="u-uext")
    print "Alpha linearization 2:\\n    %s" % symutil.linearize(s, "u", "alpha", term="u+uext")
    print "Taylor series linearization:\\n    %s" % symutil.linearize(s, "u", "taylor")

    """
    # TODO/FIXME: There Can Be Only One: maybe we should do multivar, too?

    valid_kinds_list = ["picard", "alpha", "taylor"]
    if kind not in valid_kinds_list:
        valid_kinds_str = reduce(lambda a,b: a + ", " + b, valid_kinds_list)
        raise ValueError("Kind '%s' not recognized. Valid kinds: %s. See docstring for more information." % (kind, valid_kinds_str))

    if type(expr) == str:
        expr = sy.sympify(expr)

    # generate corresponding old field name from var
    var0 = "%s0" % var
    if is_new_sympy():
        if type(var)  == str:
            var  = sy.symbols(var)
        if type(var0) == str:
            var0 = sy.symbols(var0)

    # Basic fixed point: replace u by u0.
    if kind == "picard":
        return str(expr.subs( {var : var0} ))

    # Taylor series expansion.
    elif kind == "taylor":
        # Use our taylor wrapper to work around bug in SymPy 0.6.x.
        # For SymPy 0.7.x+, it is a passthrough.
        #
        # Parameter "1" = linear (order of polynomial).
        #
        return str(taylor(expr, var, var0, 1).removeO())

    # "Alpha" is based on factorizing and extracting a linear factor. It works on polynomials.
    elif kind == "alpha":
        # TODO/FIXME: Improve this not to require user-given term, but find it automatically.
        # TODO/FIXME: Try to factor, scan the resulting expression tree and pick any
        # TODO/FIXME: linear polynomial factor?
        if "term" not in kwargs:
            raise ValueError("When kind='alpha', the kwarg 'term' is required. See docstring for more information.")
        term = kwargs["term"]

        linterm = sy.sympify(term)
        try:
            if is_new_sympy():
                d = sy.Poly(linterm, var).degree()
            else:
                d = sy.Poly(linterm, str(var)).degree

            if d != 1:
                raise ValueError("When kind='alpha', kwarg 'term' must be a linear polynomial of var (e.g. 'var - c', where c is a constant). However, term='%s' is not a linear polynomial in var='%s'. Please check." % (term, var))
        except sy.PolynomialError:
            raise ValueError("When kind='alpha', kwarg 'term' must be a linear polynomial of var (e.g. 'var - c', where c is a constant). However, term='%s' is not a linear polynomial in var='%s'. Please check." % (term, var))

        sym_expr = sy.sympify(expr)
        try: # ...interpreting sym_expr as a polynomial
            if is_new_sympy():
                p = sy.Poly(sym_expr, var)
            else:
                p = sy.Poly(sym_expr, str(var))
        except sy.PolynomialError:
            raise ValueError("When kind='alpha', expr must be a polynomial in var. However, expr='%s' is not a polynomial in var='%s'." % (expr, var))

        # Factor the polynomial
        facs = sy.factor(p)

        # Extract the coeff of linterm in p:  p = coeff*linterm.
        #
        # Note that if the constant term c in linterm is negative,
        # SymPy may write -(c - u) instead of (u - c);
        # account for this by trying both (u - c) and (c - u).
        #
        dic = sy.collect(facs, linterm, evaluate=False)
        if linterm not in dic:
            linterm = -linterm
            dic = sy.collect(facs, linterm, evaluate=False)

        if linterm not in dic:
            raise ValueError("When kind='alpha', term must be a factor of expr. However, term='%s' is not a factor of expr='%s'. Please change the term." % (term,expr))

        coeff = dic[linterm].subs( { var : var0 } )
        sym_linearized = linterm*coeff

#        # Add back the constant part if it existed
#        # (FIXME: there shouldn't be any, if linterm is a factor?)
#        SymbolOne = sy.sympify("1")
#        if SymbolOne in dic:
#            sym_linearized += dic[SymbolOne]
        SymbolOne = sy.sympify("1")
        if SymbolOne in dic:
            raise ValueError("Something went wrong: expr='%s' should divide evenly by term='%s', but got remainder='%s'." % (expr,term,str(dic[SymbolOne])) )

        return str(sym_linearized)

    # All cases covered; no "else" needed.

