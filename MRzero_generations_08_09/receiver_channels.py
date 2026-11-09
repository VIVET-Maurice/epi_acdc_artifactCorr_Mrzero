"""Validate physical receiver identifiers without inventing channel numbers."""

from __future__ import annotations

import numpy as np


def receiver_channel_ids(blocks, receiver_count, *, label='Twix'):
    """Return physical IDs in data-row order, or None when all are unknown.

    Call after loading the ADC data and checking its receiver dimension.
    Siemens VE channel headers follow the exact row order of ``block.data``;
    the identifiers need not be consecutive or begin at zero. Their order
    must remain fixed throughout a measurement, including its navigators.

    Older formats and synthetic inputs may lack channel headers entirely.
    Missing headers are permitted only when every supplied block lacks them.
    A present but incomplete/duplicate header is invalid, and mixing known
    and unknown identities is never treated as evidence of matching channels.
    """
    identities = []
    for index, block in enumerate(blocks):
        headers = getattr(block, 'channel_hdr', None)
        if headers is None:
            identities.append(None)
            continue
        try:
            count = len(headers)
        except TypeError as error:
            raise ValueError(f'{label} receiver channel headers must be a sequence.') from error
        if count != receiver_count:
            raise ValueError(f'{label} receiver channel header count disagrees with the data rows in ADC block {index}.')
        channels = []
        for header in headers:
            value = getattr(header, 'ChannelId', None)
            if (value is None or isinstance(value, (bool, np.bool_))
                    or not np.isscalar(value) or not np.isreal(value)):
                raise ValueError(f'{label} receiver channel headers need finite nonnegative integer ChannelId values.')
            try:
                numeric = float(value)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f'{label} receiver channel headers need finite nonnegative integer ChannelId values.') from error
            if not np.isfinite(numeric) or numeric < 0 or numeric != np.floor(numeric):
                raise ValueError(f'{label} receiver channel headers need finite nonnegative integer ChannelId values.')
            channels.append(int(numeric))
        if len(set(channels)) != receiver_count:
            raise ValueError(f'{label} has duplicate receiver channel IDs in ADC block {index}.')
        identities.append(channels)
    known = [value for value in identities if value is not None]
    if not known:
        return None
    if len(known) != len(identities):
        raise ValueError(f'{label} mixes known and missing receiver channel IDs across ADC blocks.')
    if any(value != known[0] for value in known[1:]):
        raise ValueError(f'{label} receiver channel IDs or their order change between ADC blocks.')
    return known[0].copy()


__all__ = ['receiver_channel_ids']
