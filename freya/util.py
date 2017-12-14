# -*- coding: utf-8 -*-
#
# Generic utility functions for Freya.
#
# JJ 2012-02-29

from __future__ import division

import re
import math

import numpy as np

import scipy.io  # savemat(), loadmat()

import matplotlib.colors

# http://stackoverflow.com/questions/1714027/version-number-comparison
def vercmp(version1, version2):
    """Compare version numbers.

    Return value:
      +1 if version1 >  version2,
       0 if version1 == version2,
      -1 if version1 <  version2.

    """
    def normalize(v):
#        return [int(x) for x in re.sub(r'(\.0+)*$','', v).split(".")]  # errors on e.g. "0.7.1.rc1"
        return [x for x in re.sub(r'(\.0+)*$','', v).split(".")]
    return cmp(normalize(version1), normalize(version2))


def prettyprint_bool(b):
    """Convert a bool to text 'on' or 'off'."""
    return "on" if b else "off"


def round_sig(x, sig=2, method='round'):
    """Round float x to sig significant figures.

    method = string:
      'round': round by standard rounding rules (default)
      'ceil':  always round up
      'floor': always round down

    See
    http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python"""
#    # As per answer by indgar, and adding abs() as suggested by dgorissen to handle also x<0.
#    # Added also handling for zero as a special case.
#    #
#    if x == 0.0:
#        return x
#    else:
#        return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

    # Chris Stringfellow's answer (which allows for generalization to "ceil"/"floor" also):
    #
    # To round an integer to 1 significant figure the basic idea is to convert it to a floating point
    # with 1 digit before the point and round that, then convert it back to its original integer size.
    #
    # To do this we need to know the largest power of 10 less than the integer. We can use floor of
    # the log 10 function for this. 
    #
    #from math import log10, floor
    #def round_int(i,places):
    #    if i == 0:
    #        return 0
    #    isign = i/abs(i)
    #    i = abs(i)
    #    if i < 1:
    #        return 0
    #    max10exp = floor(log10(i))
    #    if max10exp+1 < places:
    #        return i
    #    sig10pow = 10**(max10exp-places+1)
    #    floated = i*1.0/sig10pow
    #    defloated = round(floated)*sig10pow
    #    return int(defloated*isign)

    # Adapting the above for floats, and adding logic to round always up or down, we have:

    if x == 0.0:
        return x

    xsign = x/abs(x)
    x = abs(x)

    if xsign > 0:
        if method == 'ceil':
            rounded = math.ceil
        elif method == 'floor':
            rounded = math.floor
        else: # method == 'round':
            rounded = round
    else:
        # emulate correct behaviour of ceil/floor for negative input
        # (given that our x is always positive at this point)
        if method == 'floor':
            rounded = math.ceil
        elif method == 'ceil':
            rounded = math.floor
        else: # method == 'round':
            rounded = round

    max10exp = math.floor(math.log10(x))
    sig10pow = 10**(max10exp-sig+1)
    floated = x*1.0/sig10pow
    defloated = rounded(floated)*sig10pow
    return defloated*xsign


def sanitize_ndim(d):
    """Sanity check and sanitize the number of space dimensions.

    In: raw value (number). Out: sanitized value.
    Raises ValueError() if the given value is invalid.

    """
    d = math.floor(d)
    if d < 1  or  d > 2:
        raise ValueError("Number of dimensions out of range. Valid values are 1 and 2.")
    return d


def f5(seq, idfun=None): 
   """Uniqify a list (remove duplicates).

   This is the fast order-preserving uniqifier "f5" from
   http://www.peterbe.com/plog/uniqifiers-benchmark

   The list does not need to be sorted.

   The return value is the uniqified list.

   """
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result


def wholeword_sub(pat, rep, s):
    r"""Whole-word regex replacement for cases where \b won't cut it.

    Specifically, mathematical operators such as * are not understood
    as word boundaries by \b, so we use \W with look-ahead and look-behind,
    and do separate checks at the start of the string, at the end of the
    string, and for the whole string.

    """
    # CRUD! Any other special cases to watch for?
    temppat = re.sub( "\(", r"\(", pat )
    temppat = re.sub( "\)", r"\)", temppat )

    o = re.sub( "(?<=\W)%s(?=\W)" % temppat, rep, s ) # whole-word, middle
    o = re.sub( "^%s(?=\W)"       % temppat, rep, o ) # whole-word, start of string
    o = re.sub( "(?<=\W)%s$"      % temppat, rep, o ) # whole-word, end of string
    o = re.sub( "^%s$"            % temppat, rep, o ) # whole-word, whole string
    return o


def wholeword_find(pat, s):
    """Like wholeword_sub(), but just returns a list of matches."""
    # CRUD! Any other special cases to watch for?
    temppat = re.sub( "\(", r"\(", pat )
    temppat = re.sub( "\)", r"\)", temppat )

    m = []
    m.extend( re.findall("(?<=\W)%s(?=\W)" % temppat, s) )
    m.extend( re.findall("^%s(?=\W)"       % temppat, s) )
    m.extend( re.findall("(?<=\W)%s$"      % temppat, s) )
    m.extend( re.findall("^%s$"            % temppat, s) )
    return m


def remap(ulim, L):
    """Remap range(ulim), skipping the numbers given in the list L.

    Return the tuple (orig2remapped,remapped2orig), where each item
    is a list giving the mapping to one direction:

    orig2remapped[original] = remapped,  or -1 if original is in L
    remapped2orig[remapped] = original

    If L is the empty list, return identity mappings in both directions.

    Used for Dirichlet DOF elimination; see EquationSystem.__find_dirichlet_dofs().

    """
    if len(L) > 0:
        # Generate an identity mapping up to L[0]
        orig2remapped = []
        remapped2orig = []
        start_offs = 0
        if L[0] != 0:
            orig2remapped.extend( range(L[0]) )
            remapped2orig.extend( range(L[0]) )
            start_offs = L[0]
            orig2remapped.append( -1 ) # L[0] is in L, it has no final index
        else:
            orig2remapped.append( -1 ) # 0 is in L, it has no remapped index

        # Process the bulk
        for j in xrange(1,len(L)):
            if L[j] - L[j-1] > 1: # check presence of gap in L  =>  non-skipped numbers
                N = L[j] - L[j-1] - 1
                # index: original, data: remapped
                orig2remapped.extend( range(start_offs, start_offs + N) ) 
                start_offs += N  # this many entries were added
                # index: remapped, data: original
                remapped2orig.extend( range(L[j-1]+1, L[j]) )
            orig2remapped.append( -1 ) # j is in L, it has no final index

        # Add the remaining numbers (if any)
        if L[-1] != (ulim-1):
            N_remain = (ulim-1) - L[-1]
            orig2remapped.extend( range(start_offs, start_offs + N_remain) )
            remapped2orig.extend( range(L[-1]+1, ulim) )
        else:
            # FIXME: WTF is going on here? Sometimes we need to add the
            # FIXME: final element here and sometimes not. Hacking it for now.
            if len(orig2remapped) < ulim:
                orig2remapped.append( -1 )  # last index is in L
    else:
        # L is empty; set up identity mappings
        orig2remapped = range(ulim)
        remapped2orig = range(ulim)

    # Each number in the range either gets a new number or -1.
    # Hence the length does not change. Any numbers missing or added => bug.
    assert( len(orig2remapped) == ulim )

    return (orig2remapped, remapped2orig)


def load_balance_list(L, nprocs):
    """Given a list of arbitrary items, split it to nprocs roughly equal-sized parts.

    This is useful for dividing a list of work items in MPI parallelization.
    It is assumed that each work item takes the same amount of time; hence the
    initial distribution is generated by naive integer division.

    If len(L) does not divide evenly with nprocs, the remaining items are distributed
    on an item-by-item basis to the first (len(L) mod nprocs) processes.

    If nprocs > len(L), the items will be distributed on an item-by-item basis to the
    first len(L) processes, and the rest of the processes will get an empty list.
    
    Examples:
        load_balance_list(range(14), 2)
            =>  [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]]
        In this case, the division is even (no remainder).

        load_balance_list(range(15), 2)
            =>  [[0, 1, 2, 3, 4, 5, 6, 14], [7, 8, 9, 10, 11, 12, 13]]
        In this case, the item "14" is left over after the integer division.
        The leftover item is given to the first process.

        load_balance_list(range(4), 8)
            =>  [[0], [1], [2], [3], [], [], [], []]
        In this case, nprocs is so large that there are not enough items to give
        even one item to each of them. The empty list is given for those for which
        no item is available.

    Parameters:
        L = any Python list.

    Return value:
        List of lists: L_out = [L1, L2, L3, ..., Lnprocs]
        where L1, L2, ... are sublists of L.

        It always holds that len(L_out) == nprocs. Note that the lengths of the
        Lj (individual lists inside L_out) may differ by one item, depending on
        whether the division was even.

        If nprocs == 1, the return value is [L] for compatibility
        of the output format with the nprocs > 1 case.

    """
    # If we're "splitting" to one part, this is a no-op.
    # But wrap it - the caller is expecting a list of lists.
    #
    if nprocs == 1:
        return [L]

    ntasks    = len(L)  # number of task items that need processing

    out = []
    if nprocs <= ntasks:
        # Items per process.
        blocklen  = ntasks // nprocs  # integer division!

        # Leftover items.
        remainder = ntasks - blocklen*nprocs  # this is always < nprocs

        # Distribute the items that divided evenly.
        for m in xrange(nprocs):
            offs = m*blocklen
            out.append( L[offs:offs+blocklen] )

        # Distribute the leftovers (if any).
        if remainder > 0:
            offs = ntasks - remainder
            for m in xrange(remainder):
                out[m].append( L[offs] )
                offs += 1
    else:
        # Distribute one item to each process as long as items are available.
        for m in xrange(ntasks):
            out.append( [L[m]] )  # wrap the item to make a single-item list.

        # Give an empty list to the rest of the processes.
        nempties = nprocs - ntasks  # this many empty lists are needed
        for m in xrange(nempties):
            out.append( [] )

    assert( len(out) == nprocs )
    return out


def flatten_list_of_arrays( data, datatype ):
    """Concatenate Python list of rank-n np.arrays with the same dtype (and rank!) into a single rank-n np.array.

    The datatype must be provided, because any of the arrays are allowed to be empty (0 elements).

    Optionally, any of the items are allowed to be the empty Python list; this is equivalent to the
    0-element rank-n np.array.

    """
    # number of "items":
    #    
    # - the empty list has zero items
    # - a scalar (variable of a primitive datatype, e.g. int) has one item
    # - an np.array has as many items as its first dimension
    #
    def num_items(obj):
        s = np.shape(obj)
        if len(s) > 0:
            return s[0]
        else:  # np.shape(my_scalar_variable) ->  ()
            return 1

    # Compute total number of elements.
    #
    # Note that np.shape( [] ) == (0,), so this also works if some of the inputs are empty *lists*.
    #
    total_len = np.sum( map( num_items, data ) )

    if total_len == 0:
        return np.empty( [0], dtype=datatype )

    # find first non-empty array in data. This is guaranteed to succeed because here total_len > 0.
    for k in xrange(len(data)):
        if num_items(data[k]) != 0:
            break

    # new shape: total_len by rest-of-dimensions-as-is
    #
    oldshape = np.shape(data[k])
    newshape = [total_len]
    newshape.extend( oldshape[1:] )

    result = np.empty( newshape, dtype=datatype )
    view_out = np.reshape( result, (-1,) )

    offs_out = 0
    for j in xrange(len(data)):
        view_in = np.reshape( data[j], (-1,) )  # this casts also scalars into arrays

        item_len = np.size(data[j])
        view_out[offs_out:(offs_out+item_len)] = view_in
        offs_out += item_len
    return result


def strip_subscript(string):
    """Given a string, strip everything from the first "_" onward (inclusive)."""
    if len(re.findall(r'_', string)) > 0:  # has a subscript?
        string = re.sub(r'_.*$', r'', string)
    return string


def make_colormap(seq, name):
    """Return a LinearSegmentedColormap

    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    name: string, a name for the new colormap.

    Usage example:

    c = matplotlib.colors.ColorConverter().to_rgb
    rvb = make_colormap( [c('red'), c('violet'), 0.33, c('violet'), c('blue'), 0.66, c('blue')] )

    Or, equivalently:

    rvb = make_colormap( [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0), 0.33, (1.0, 0.0, 1.0), (0.0, 0.0, 1.0), 0.66, (0.0, 0.0, 1.0)] )

    First color is always taken at the beginning of the scale (value 0.0).
    This example fades from red (at 0.0) to violet at 0.33; then from violet at 0.33
    to blue at 0.66, and then remains blue until the end of the scale (value 1.0).

    The implementation comes from

    http://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale

    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return matplotlib.colors.LinearSegmentedColormap(name, cdict)        


def remap_colors(cmap_name_orig, cmap_name_new, new_stops):
    """Remap pre-existing colormap cmap_name_orig using new custom stops.

    Number of stops to use is determined automatically from the length of new_stops.

    The resulting modified colormap is registered in the running
    Matplotlib instance as cmap_name_new.

    This is useful for making non-linearly progressing versions
    of the Matplotlib built-in colormaps.

    new_stops: rank-1 np.array of floats in increasing order
       new_stops[0]  must be 0.0
       new_stops[-1] must be 1.0

    Usage example:

    newstops = numpy.linspace(0.0, 1.0, 21)**2  # emphasize low end (allocate more colors for small values)
    remap_colors("Oranges", "Oranges_custom", newstops)
    pl.pcolor( X,Y,Z, cmap="Oranges_custom", edgecolors='none' )
    pl.colorbar()

    This comes from

    http://matplotlib.1069221.n5.nabble.com/get-colorlist-and-values-from-existing-matplotlib-colormaps-td23788.html

    See

    http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps

    for a recipe to find out what colormaps your installation of Matplotlib has available.

    """
    # http://matplotlib.1069221.n5.nabble.com/get-colorlist-and-values-from-existing-matplotlib-colormaps-td23788.html
    ncolors = len(new_stops)
    o_cmap = pl.cm.get_cmap(cmap_name_orig, ncolors)
    o_vals = o_cmap(np.arange(ncolors))[:,:-1]  # drop alpha values

    seq = []
    seq.append( o_vals[0,:].tolist() )
    for j in xrange(1,ncolors-1):
        seq.append( o_vals[j,:].tolist() )
        seq.append( new_stops[j] )
    seq.append( o_vals[-1,:].tolist() )

    custom_cmap = make_colormap(seq, cmap_name_new)
    pl.cm.register_cmap(name=custom_cmap.name, cmap=custom_cmap)


def mark_colorlevels(cbar, lvs):
    """Mark given color levels lvs in the colorbar cbar by black horizontal lines.

    cbar = colorbar instance
    lvs  = levels to mark; np.array of floats in [0.0, 1.0]

    Usage example:

    cbar = pylab.colorbar()
    mark_colorlevels( cbar, np.array( [0.25, 0.5] ) )  # something important occurs at data values 0.25 and 0.5


    """
    # The API is extremely general, but a bit of a nightmare for simple cases like this;
    # first we must create a ContourSet describing the geometry of the lines we want.
    #
    # See
    # http://matplotlib.org/examples/misc/contour_manual.html
    # http://fossies.org/dox/matplotlib-1.3.1/classmatplotlib_1_1contour_1_1ContourSet.html

    segs = []
    for j in xrange(np.size(lvs)):
        # Make a horizontal line across the whole colorbar (from x=0.0 to x=1.0).
        #
        # The y coordinates of the segments here seem to be relative to the "levels"
        # argument given to the constructor of ContourSet; since we set the actual values
        # there, here the y coordinates must be set to zero.
        #
        segs.append( [ np.array( [[0.0,0.0], [1.0,0.0]] ) ] )

    # Make the lines black. For this, we need to make a custom cmap, as it is not provided by default.
    cseq = []
    cseq.append( [0.0, 0.0, 0.0] )
    cseq.append( [0.0, 0.0, 0.0] )
    just_black = make_colormap(cseq, "just_black")

    # Now we can create the ContourSet, and add it to the ColorBar.
    #
    # Note that the first three parameters (ax, levels, allsegs) must be given unnamed;
    # otherwise they will end up in kwargs and the constructor will not interpret them correctly.
    #
    import matplotlib.contour

    # Version for histogram equalizer, where f(p) is not linear on the colorbar axis;
    # a priori, we have no idea at which height different data values have ended up at
    # (since it depends on the data itself).
    #
    # Hence to find the height, on the colorbar, of a given data value x, we must compute f(x).

    # force list even for a single value
    #
    # NOTE: if the data is somehow mogrified (e.g. by a histogram remapper), we should adjust the fvalues here.
    # See miniprojects/misc/rollnDd.py (and search for "just_black") for an example.
    #
    fvalues = lvs.tolist()
    if type(fvalues) != list:
        fvalues = [fvalues]

    CS = matplotlib.contour.ContourSet( cbar.ax, fvalues, segs, cmap=just_black )
    cbar.add_lines( CS )


def relabel_cticks(cbar, fmt, func):
    """Relabel colorbar ticks, using printf format fmt (e.g. "%0.2g"), optionally mapping original values through function func.

    If func is None, just re-format existing values using format fmt.

    See

    http://matplotlib.org/examples/pylab_examples/colorbar_tick_labelling_demo.html

    """
    cticks = cbar.ax.get_yticks()

    labels = []
    if f is not None:
        for value in cticks:
            labels.append( fmt % f(value) )
    else:
        for value in cticks:
            labels.append( fmt % value )
    return labels

    cbar.ax.set_yticklabels( labels )


def prettify_colorbar(cbar, **kwargs):
    """Prettify the colorbar cbar.

    cbar = colorbar instance as returned by pylab.colorbar().

    This e.g. ensures that the actual min/max value is labeled,
    and rounds tick labels to a sensible display precision.

    Kwargs:
        relabel = integer or None. If integer, relabel colorbar with
                  this many ticks, spaced linearly.

    """
    if "relabel" in kwargs  and  kwargs["relabel"] is not None:
        N = kwargs["relabel"]
    else:
        N = None

    if N is None:
        # We label also the endpoints, even though it's not a nice round number.
        # We would like to keep all existing ticks, and just add these ones.
        #
        # First, get the existing ticks:
        #
        cticks = cbar.ax.get_yticks().tolist()  # these are in range [0,1] (as fraction of colorbar height)

        # Remove any ticks too near the ends to avoid overlap with min/max labels
        #
        # TODO: should compute font height here, but in general it's not available before the figure
        # TODO: is actually blitted for the first time.
        #
        eps = 0.03
        cticks = filter( lambda x: x > eps and x < 1.0 - eps,  cticks )

        # Add min/max ticks.
        #
        cticks = [0.0] + cticks + [1.0]
    else:
        cticks = np.linspace(0.0, 1.0, N)

    # We must now jump through an extra hoop, because cbar.ax.set_yticks() does nothing except
    # printing a warning that one should use cbar.set_ticks() instead. However, the scaling of
    # cbar.set_ticks() is different from the one used by cbar.ax.get_yticks()!
    #    
    # We must thus first convert the colorbar height fraction values (which is what we have)
    # into actual data values (which is what cbar.set_ticks() expects).
    #
    cmin,cmax = cbar.get_clim()
    tickvalues = cmin + np.array(cticks)*(cmax - cmin)

    cbar.set_ticks( tickvalues )

    # For labels only: round off the tick values for a cleaner look
    #
    ticklabelvalues = map( lambda x: round_sig(x, 3), tickvalues )

    # Round label to zero if value "near enough" (absolute value less than 1.5% of data range).
    # This prevents ugliness such as 1, 0.8, ..., 0.2, 1.3e-6.
    #
    ticklabelvalues = map( lambda x: x if abs(x) >= 0.015 * (cmax - cmin) else 0.0, ticklabelvalues )

    # Finally, eliminate extra zeroes from round numbers to avoid labels like "0.0".
    #
    # http://stackoverflow.com/questions/2440692/formatting-floats-in-python-without-superfluous-zeros
    cbar.ax.set_yticklabels( [('%f' % value).rstrip('0').rstrip('.') for value in ticklabelvalues] )

#    # Align numbers as right-justified
#    #
#    # http://stackoverflow.com/questions/19219963/align-ticklabels-in-matplotlib-colorbar
#    for t in cbar.ax.get_yticklabels():
#        t.set_horizontalalignment('right')   
#        t.set_x(3.5)

    return cbar


def determine_clim(data, **kwargs):
    """Determine plotting colour limits.

    Default is min and max of data (np.array, any shape).

    Optionally, cutting can be applied to handle data where a very small (uninteresting)
    region covers most of the value range (such as when there are corner singularities).

    Kwargs:
        cut_low  = float, (0,1).
                   Cut away this much data mass (based on histogram, automatically computed)
                   at the low end.

                   E.g. cut_low=0.05 ignores 5% of data mass having the smallest values.

        cut_high = float, (0,1).
                   Cut away this much data mass (based on histogram, automatically computed)
                   at the high end.

                   E.g. cut_high=0.05 ignores 5% of data mass having the largest values.

    Return value:
        tuple (cmin, cmax), where cmin and cmax are floats representing the colour limits;
        the values have the same scaling as the input data (so that they can be directly used
        in plotting commands).

    """
    if "cut_low" in kwargs:
        cut_low = kwargs["cut_low"]
    else:
        cut_low = None

    if "cut_high" in kwargs:
        cut_high = kwargs["cut_high"]
    else:
        cut_high = None

    u = np.reshape(data, -1)

    # Don't bother with making a histogram if no cutting is applied
    #
    if cut_low is None  and  cut_high is None:
        return (np.min(u), np.max(u))

    # Using a histogram, find cut thresholds based on data mass.
    #
    H,bin_edges = np.histogram( u, bins=128 )
    Hf = np.array(H, dtype=np.float64)
    cdf = np.cumsum(Hf)
    cdfmax = np.max(cdf)
    Hf  /= cdfmax  # normalize sum to 1
    cdf /= cdfmax  # normalize maximum to 1

    if cut_low is None:
        cmin = np.min(u)
    else:
        # find first bin number where the cumulative distribution function exceeds cut_low
        cut_index = np.min( (cdf > cut_low).nonzero()[0] )
        cmin = bin_edges[cut_index]  # take lower edge of the bin as cmin

    if cut_high is None:
        cmax = np.max(u)
    else:
        # find first bin number where the cumulative distribution function exceeds 1 - cut_high
        # (we use >= to always get at least one match even if cut_high = 0;
        #  then the last bin matches)
        cut_index = np.min( (cdf >= (1.0 - cut_high)).nonzero()[0] )
        cmax = bin_edges[cut_index+1]  # take upper edge of the bin as cmax

    return (cmin, cmax)


def round_clim(cmin, cmax):
    """Prettify colour limits by rounding them to two significant digits.

    In addition, if either limit "clearly wants to be zero" (has absolute value
    less than 1.5% of data range), it is zeroed.

    Note that in order to avoid unwanted holes in the plot near the extreme values,
    this always rounds "away from data" (i.e. min is always rounded down, max up).

    Return value:
        tuple (cmin, cmax) with updated values

    """
    roundmin = round_sig(cmin, 2, method='floor')
    roundmax = round_sig(cmax, 2, method='ceil')

    # Special handling for zero min/max: round to zero if the absolute value of data limit
    # is less than 1.5% of data range
    #
    eps = 0.015 * (roundmax - roundmin)
    if abs(roundmin) < eps:
        roundmin = 0.0
    if abs(roundmax) < eps:
        roundmax = 0.0

    return (roundmin,roundmax)


def save_to_mat(filename, varnames, context):
    """Save variables in a MATLAB-compatible .mat file.

    This is a wrapper for SciPy's io, setting some default parameters
    (format='5', oned_as='row') and automatically naming the
    saved variables in the .mat file.

    Parameters:
        filename = string. Name of .mat file to save to.
        varnames = list of strings. Names of the variables to be saved.
        context  = dictionary in which to look up varnames.
                   Pass in locals() here to do the matlab-ish intuitive thing.

    Return value:
        No return value.

    Examples:
        import numpy as np
        a = np.random.random( (5,5) )
        b = np.random.random( (3) )
        save_to_mat( "my_file.mat", ["a", "b"], locals() )

        # now my_file.mat contains arrays named "a" and "b"

        # also ok if saving only one variable:
        save_to_mat( "a_only.mat", "a", locals() )

        # save variables contained in a dictionary
        c = { "a" : a, "b" : b }
        save_to_mat( "bothagain.mat", c.keys(), c )

    """
    if type(varnames) == str:
        varnames = [varnames]
    data = dict([(name,context[name]) for name in varnames])
    scipy.io.savemat(filename, data, format='5', oned_as='row')


def load_from_mat(filename, varnames):
    """Load variables from a MATLAB-compatible .mat file.

    This is a wrapper for SciPy's io, filtering out the MATLAB headers
    from the returned dictionary.

    Parameters:
        filename = string. Name of .mat file to load.
        varnames = list of strings. Names of the variables to be loaded.
                   Can be None, which means "load all" (as in SciPy's io).

    Return value:
        dict. Sanitized output in the form  varname : value.

    """
    if varnames is not None:
        varnames = list(varnames)  # copy; loadmat mangles the list passed in

    d = scipy.io.loadmat(filename, variable_names=varnames)

    # Filter out the headers (they have __ at the beginning of the key)
    d = dict( filter( lambda item: not item[0].startswith("__"), d.items() ) )
    
    return d

