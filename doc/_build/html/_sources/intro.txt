Workflow
************

1. Use class :py:class:`watershed3d.Ws3d`

.. 2. Assemble .h5 file needed by Ilastik using :py:func:`fileutil.make_h5`,
.. 3. Run Ilastik and export tracking information as series of .h5 files
.. 4. Assemble trajectories using :py:func:`ilread.assemble_trajectories`
.. 5. Assemble cell object and outline information using :py:func:`ilread.all_object_indices`
.. 6. Check by compiling a movie: :py:func:`ilread.compile_movie`