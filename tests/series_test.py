import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak


class SeriesTest(ArkoudaTest):
    def test_series_creation(self):
        # Use positional arguments
        ar_tuple = ak.arange(3), ak.arange(3)
        s = ak.Series(ar_tuple)
        self.assertIsInstance(s, ak.Series)

        ar_tuple = ak.array(["A", "B", "C"]), ak.arange(3)
        s = ak.Series(ar_tuple)
        self.assertIsInstance(s, ak.Series)

        # Both data and index are supplied
        v = ak.array(["A", "B", "C"])
        i = ak.arange(3)
        s = ak.Series(data=v, index=i)

        self.assertIsInstance(s, ak.Series)
        self.assertIsInstance(s.index, ak.Index)

        # Just data is supplied
        s = ak.Series(data=v)
        self.assertIsInstance(s, ak.Series)
        self.assertIsInstance(s.index, ak.Index)

        # Just index is supplied (keyword argument)
        with self.assertRaises(TypeError):
            s = ak.Series(index=i)

        # Just data is supplied (positional argument)
        s = ak.Series(ak.array(["A", "B", "C"]))
        self.assertIsInstance(s, ak.Series)

        # Just index is supplied (ar_tuple argument)
        ar_tuple = (ak.arange(3),)
        with self.assertRaises(TypeError):
            s = ak.Series(ar_tuple)

        # No arguments are supplied
        with self.assertRaises(TypeError):
            s = ak.Series()

        with self.assertRaises(ValueError):
            s = ak.Series(data=ak.arange(3), index=ak.arange(6))

    def test_lookup(self):
        v = ak.array(["A", "B", "C"])
        i = ak.arange(3)
        s = ak.Series(data=v, index=i)

        lk = s.locate(1)
        self.assertIsInstance(lk, ak.Series)
        self.assertEqual(lk.index[0], 1)
        self.assertEqual(lk.values[0], "B")

        lk = s.locate([0, 2])
        self.assertIsInstance(lk, ak.Series)
        self.assertEqual(lk.index[0], 0)
        self.assertEqual(lk.values[0], "A")
        self.assertEqual(lk.index[1], 2)
        self.assertEqual(lk.values[1], "C")

        # testing index lookup
        i = ak.Index([1])
        lk = s.locate(i)
        self.assertIsInstance(lk, ak.Series)
        self.assertListEqual(lk.index.to_list(), i.index.to_list())
        self.assertEqual(lk.values[0], v[1])

        i = ak.Index([0, 2])
        lk = s.locate(i)
        self.assertIsInstance(lk, ak.Series)
        self.assertListEqual(lk.index.to_list(), i.index.to_list())
        self.assertEqual(lk.values.to_list(), v[ak.array([0, 2])].to_list())

        # testing multi-index lookup
        mi = ak.MultiIndex([ak.arange(3), ak.array([2, 1, 0])])
        s = ak.Series(data=v, index=mi)
        lk = s.locate(mi[0])
        self.assertIsInstance(lk, ak.Series)
        self.assertListEqual(lk.index.index, mi[0].index)
        self.assertEqual(lk.values[0], v[0])

        # ensure error with scalar and multi-index
        with self.assertRaises(TypeError):
            lk = s.locate(0)

        with self.assertRaises(TypeError):
            lk = s.locate([0, 2])

    def test_shape(self):
        v = ak.array(["A", "B", "C"])
        i = ak.arange(3)
        s = ak.Series(data=v, index=i)

        (l,) = s.shape
        self.assertEqual(l, 3)

    def test_add(self):
        i = ak.arange(3)
        v = ak.arange(3, 6, 1)
        s = ak.Series(data=i, index=i)

        s_add = ak.Series(data=v, index=v)

        added = s.add(s_add)

        idx_list = added.index.to_pandas().tolist()
        val_list = added.values.to_list()
        for i in range(6):
            self.assertIn(i, idx_list)
            self.assertIn(i, val_list)

    def test_topn(self):
        v = ak.arange(3)
        i = ak.arange(3)
        s = ak.Series(data=v, index=i)

        top = s.topn(2)
        self.assertListEqual(top.index.to_pandas().tolist(), [2, 1])
        self.assertListEqual(top.values.to_list(), [2, 1])

    def test_sort_idx(self):
        v = ak.arange(5)
        i = ak.array([3, 1, 4, 0, 2])
        s = ak.Series(data=v, index=i)

        sorted = s.sort_index()
        self.assertListEqual(sorted.index.to_pandas().tolist(), [i for i in range(5)])
        self.assertListEqual(sorted.values.to_list(), [3, 1, 4, 0, 2])

    def test_sort_value(self):
        v = ak.array([3, 1, 4, 0, 2])
        i = ak.arange(5)
        s = ak.Series(data=v, index=i)

        sorted = s.sort_values()
        self.assertListEqual(sorted.index.to_pandas().tolist(), [3, 1, 4, 0, 2])
        self.assertListEqual(sorted.values.to_list(), [i for i in range(5)])

    def test_head_tail(self):
        v = ak.arange(5)
        i = ak.arange(5)
        s = ak.Series(data=v, index=i)

        head = s.head(2)
        self.assertListEqual(head.index.to_pandas().tolist(), [0, 1])
        self.assertListEqual(head.values.to_list(), [0, 1])

        tail = s.tail(3)
        self.assertListEqual(tail.index.to_pandas().tolist(), [2, 3, 4])
        self.assertListEqual(tail.values.to_list(), [2, 3, 4])

    def test_value_counts(self):
        v = ak.array([0, 0, 1, 2, 2])
        i = ak.arange(5)
        s = ak.Series(data=v, index=i)

        c = s.value_counts()
        self.assertListEqual(c.index.to_pandas().tolist(), [0, 2, 1])
        self.assertListEqual(c.values.to_list(), [2, 2, 1])

        c = s.value_counts(sort=True)
        self.assertListEqual(c.index.to_pandas().tolist(), [0, 2, 1])
        self.assertListEqual(c.values.to_list(), [2, 2, 1])

    def test_concat(self):
        v = ak.arange(5)
        i = ak.arange(5)
        s = ak.Series(data=v, index=i)

        v = ak.arange(5, 11, 1)
        i = ak.arange(5, 11, 1)
        s2 = ak.Series(data=v, index=i)

        c = ak.Series.concat([s, s2])
        self.assertListEqual(c.index.to_pandas().tolist(), [i for i in range(11)])
        self.assertListEqual(c.values.to_list(), [i for i in range(11)])

        df = ak.Series.concat([s, s2], axis=1)
        self.assertIsInstance(df, ak.DataFrame)

        ref_df = pd.DataFrame(
            {
                "idx": [i for i in range(11)],
                "val_0": [0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
                "val_1": [0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10],
            }
        )
        self.assertTrue(((ref_df == df.to_pandas()).all()).all())

    def test_pdconcat(self):
        v = ak.arange(5)
        i = ak.arange(5)
        s = ak.Series(data=v, index=i)

        v = ak.arange(5, 11, 1)
        i = ak.arange(5, 11, 1)
        s2 = ak.Series(data=v, index=i)

        c = ak.Series.pdconcat([s, s2])
        self.assertIsInstance(c, pd.Series)
        self.assertListEqual(c.index.tolist(), [i for i in range(11)])
        self.assertListEqual(c.values.tolist(), [i for i in range(11)])

        v = ak.arange(5, 10, 1)
        i = ak.arange(5, 10, 1)
        s2 = ak.Series(data=v, index=i)

        df = ak.Series.pdconcat([s, s2], axis=1)
        self.assertIsInstance(df, pd.DataFrame)

        ref_df = pd.DataFrame({0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]})
        self.assertTrue((ref_df == df).all().all())

    def test_index_as_index_compat(self):
        # added to validate functionality for issue #1506
        df = ak.DataFrame({"a": ak.arange(10), "b": ak.arange(10), "c": ak.arange(10)})
        g = df.groupby(["a", "b"])
        series = ak.Series(data=g.sum("c")["c"], index=g.sum("c").index)
        g.broadcast(series)

    def test_getitem_scalars(self):
        ints = ak.array([0,1,3,7,3])
        floats = ak.array([0.0,1.5,0.5,1.5,-1.0])
        strings = ak.array(['A','C','DE','C','Z'])

        s1 = ak.Series(index=strings, data=floats)
        with self.assertRaises(TypeError):
            s1[1.0]
        with self.assertRaises(TypeError):
            s1[1]
        s1_a1 = s1['A']
        self.assertTrue(isinstance(s1_a1, ak.Series))
        self.assertListEqual(s1_a1.index.to_list(), ['A'])
        self.assertListEqual(s1_a1.values.to_list(), [0.0])
        s1_a2 = s1['C']
        self.assertTrue(isinstance(s1_a2, ak.Series))
        self.assertListEqual(s1_a2.index.to_list(), ['C','C'])
        self.assertListEqual(s1_a2.values.to_list(), [1.5,1.5])

        s2 = ak.Series(index=ints,data=strings)
        with self.assertRaises(TypeError):
            s2[1.0]
        with self.assertRaises(TypeError):
            s2['A']
        s2_a1 = s2[7]
        self.assertTrue(isinstance(s2_a1, ak.Series))
        self.assertListEqual(s2_a1.index.to_list(), [7])
        self.assertListEqual(s2_a1.values.to_list(), ['C'])

        s2_a2 = s2[3]
        self.assertTrue(isinstance(s2_a2, ak.Series))
        self.assertListEqual(s2_a2.index.to_list(), [3,3])
        self.assertListEqual(s2_a2.values.to_list(), ['DE', 'Z'])

        s3 = ak.Series(index=floats, data=ints)
        with self.assertRaises(TypeError):
            s3[1]
        with self.assertRaises(TypeError):
            s3['A']
        s3_a1 = s3[0.0]
        self.assertTrue(isinstance(s3_a1, ak.Series))
        self.assertListEqual(s3_a1.index.to_list(), [0.0])
        self.assertListEqual(s3_a1.values.to_list(), [0])

        s3_a2 = s3[1.5]
        self.assertTrue(isinstance(s3_a2, ak.Series))
        self.assertListEqual(s3_a2.index.to_list(), [1.5,1.5])
        self.assertListEqual(s3_a2.values.to_list(), [1,7])

    def test_getitem_vectors(self):
        ints = ak.array([0,1,3,7,3])
        floats = ak.array([0.0,1.5,0.5,1.5,-1.0])
        strings = ak.array(['A','C','DE','C','Z'])

        s1 = ak.Series(index=strings, data=floats)
        with self.assertRaises(TypeError):
            s1[[1.0,2.0]]
        with self.assertRaises(TypeError):
            s1[[1,2]]
        with self.assertRaises(TypeError):
            s1[ak.array([1.0,2.0])]
        with self.assertRaises(TypeError):
            s1[ak.array([1,2])]
        for vector in [['A','Z'], ak.array(['A','Z']), ('A','Z')]:
            s1_a1 = s1[vector]
            self.assertTrue(isinstance(s1_a1, ak.Series))
            self.assertListEqual(s1_a1.index.to_list(), ['A','Z'])
            self.assertListEqual(s1_a1.values.to_list(), [0.0,-1.0])

        s1_a2 = s1['C','DE']
        self.assertTrue(isinstance(s1_a2, ak.Series))
        self.assertListEqual(s1_a2.index.to_list(), ['C','DE','C'])
        self.assertListEqual(s1_a2.values.to_list(), [1.5,0.5,1.5])

        s1_a3 = s1[[True,False,True,False,False]]
        self.assertTrue(isinstance(s1_a3, ak.Series))
        self.assertListEqual(s1_a3.index.to_list(), ['A','DE'])
        self.assertListEqual(s1_a3.values.to_list(), [0.0,0.5])

        s2 = ak.Series(index=floats, data=ints)
        with self.assertRaises(TypeError):
            s2[['A']]
        with self.assertRaises(TypeError):
            s2[[1,2]]
        with self.assertRaises(TypeError):
            s2[ak.array(['A','B'])]
        with self.assertRaises(TypeError):
            s2[ak.array([1,2])]
        for vector in [[0.5,0.0], ak.array([0.5,0.0]), (0.5,0.0)]:
            s2_a1 = s2[vector]
            self.assertTrue(isinstance(s2_a1, ak.Series))
            self.assertListEqual(s2_a1.index.to_list(), [0.0,0.5])
            self.assertListEqual(s2_a1.values.to_list(), [0,3])

        s2_a2 = s2[1.5,1.2]
        self.assertTrue(isinstance(s2_a2, ak.Series))
        self.assertListEqual(s2_a2.index.to_list(), [1.5,1.5])
        self.assertListEqual(s2_a2.values.to_list(), [1,7])

    def test_setitem_scalars(self):
        ints = [0,1,3,7,3]
        floats = [0.0,1.5,0.5,1.5,-1.0]
        strings = ['A','C','DE','C','Z']
        
        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))
        with self.assertRaises(TypeError):
            s1[1.0] = 1.0
        with self.assertRaises(TypeError):
            s1[1] = 1.0
        with self.assertRaises(TypeError):
            s1['A'] = 1
        with self.assertRaises(TypeError):
            s1['A'] = 'C'
        
        s1['A'] = 0.2
        self.assertListEqual(s1.values.to_list(), [0.2,1.5,0.5,1.5,-1.0])
        s1['C'] = 1.2
        self.assertListEqual(s1.values.to_list(), [0.2,1.2,0.5,1.2,-1.0])
        s1['X'] = 0.0
        self.assertListEqual(s1.index.to_list(), ['A','C','DE','C','Z', 'X'])
        self.assertListEqual(s1.values.to_list(), [0.2,1.2,0.5,1.2,-1.0,0.0])
        s1['C'] = [0.3, 0.4]
        self.assertListEqual(s1.values.to_list(), [0.2,0.3,0.5,0.4,-1.0,0.0])
        with self.assertRaises(ValueError):
            s1['C'] = [0.4, 0.3, 0.2]
        
        #cannot assign to Strings
        s2 = ak.Series(index=ak.array(ints), data=ak.array(strings))
        with self.assertRaises(TypeError):
            s2[1.0] = 'D'
        with self.assertRaises(TypeError):
            s2['C'] = 'E'
        with self.assertRaises(TypeError):
            s2[0] = 1.0
        with self.assertRaises(TypeError):
            s2[0] = 1
        with self.assertRaises(TypeError):
            s2[7] = 'L'
        with self.assertRaises(TypeError):
            s2[3] = ['X1', 'X2']

       
        s3 = ak.Series(index=ak.array(floats), data=ak.array(ints)) 
        self.assertListEqual(s3.values.to_list(), [0,1,3,7,3])
        self.assertListEqual(s3.index.to_list(), [0.0,1.5,0.5,1.5,-1.0])
        s3[0.0] = 2
        self.assertListEqual(s3.values.to_list(), [2,1,3,7,3])
        s3[1.5] = 8
        self.assertListEqual(s3.values.to_list(),  [2,8,3,8,3])
        s3[2.0] = 9
        self.assertListEqual(s3.index.to_list(), [0.0,1.5,0.5,1.5,-1.0,2.0])
        self.assertListEqual(s3.values.to_list(), [2,8,3,8,3,9])
        s3[1.5] = [4,5]
        self.assertListEqual(s3.values.to_list(), [2,4,3,5,3,9])
        s3[1.5] = ak.array([6,7])
        self.assertListEqual(s3.values.to_list(), [2,6,3,7,3,9])
        s3[1.5] = [8]
        self.assertListEqual(s3.values.to_list(), [2,8,3,8,3,9])
        with self.assertRaises(ValueError):
            s3[1.5] = [9,10,11]
        with self.assertRaises(ValueError):
            s3[1.5] = ak.array([0,1,2])
        with self.assertRaises(ValueError):
            s3[1.5] = ak.array([0])

    def test_setitem_vectors(self):
        ints = [0,1,3,7,3]
        floats = [0.0,1.5,0.5,1.5,-1.0]
        strings = ['A','C','DE','C','Z']

        s1 = ak.Series(index=ak.array(strings), data=ak.array(floats))
        with self.assertRaises(TypeError):
            s1[[0.1,0.2]] = 1.0
        with self.assertRaises(TypeError):
            s1[[0,3]] = 1.0
        with self.assertRaises(TypeError):
            s1[ak.array([0,3])] = 1.0
        with self.assertRaises(TypeError):
            s1[['A','B']] = 1
        with self.assertRaises(TypeError):
            s1[['A','B']] = 'C'
        with self.assertRaises(TypeError):
            s1[ak.array(['A','B'])] = 1
        
        s1[['A','Z']] = 2.0
        self.assertListEqual(s1.values.to_list(), [2.0,1.5,0.5,1.5,2.0])
        s1[ak.array(['A','Z'])] = 3.0
        self.assertListEqual(s1.values.to_list(), [3.0,1.5,0.5,1.5,3.0])
        s1[ak.array(['B','D'])] = 0.3
        self.assertListEqual(s1.values.to_list(), [3.0,1.5,0.5,1.5,3.0, 0.3, 0.3])
        self.assertListEqual(s1.index.to_list(), ['A','C','DE','C','Z', 'B','D'])
        s1[ak.array(['A','C'])] = [1.0,1.1,1.2]
        self.assertListEqual(s1.values.to_list(), [1.0,1.1,0.5,1.2,3.0, 0.3, 0.3])
        s1[('B', 'D')] = [8.0]
        self.assertListEqual(s1.values.to_list(), [1.0,1.1,0.5,1.2,3.0,8.0,8.0])
        with self.assertRaises(ValueError):
            s1[['B', 'D']] = [8.0,1.0,1.0]
        

    def test_loc(self):
        ints = ak.array([0,1,3,7,3])
        floats = ak.array([0.0,1.5,0.5,1.5,-1.0])
        strings = ak.array(['A','C','DE','C','Z'])

        s1 = ak.Series(index=strings, data=floats)
        with self.assertRaises(TypeError):
            s1.loc[1.0]
        with self.assertRaises(TypeError):
            s1.loc[1]
        s1_a1 = s1.loc['A']
        self.assertTrue(isinstance(s1_a1, ak.Series))
        self.assertListEqual(s1_a1.index.to_list(), ['A'])
        self.assertListEqual(s1_a1.values.to_list(), [0.0])
        s1_a2 = s1.loc['C']
        self.assertTrue(isinstance(s1_a2, ak.Series))
        self.assertListEqual(s1_a2.index.to_list(), ['C','C'])
        self.assertListEqual(s1_a2.values.to_list(), [1.5,1.5])

        s2 = ak.Series(index=ints,data=strings)
        with self.assertRaises(TypeError):
            s2.loc[1.0]
        with self.assertRaises(TypeError):
            s2.loc['A']
        s2_a1 = s2.loc[7]
        self.assertTrue(isinstance(s2_a1, ak.Series))
        self.assertListEqual(s2_a1.index.to_list(), [7])
        self.assertListEqual(s2_a1.values.to_list(), ['C'])

        s2_a2 = s2.loc[3]
        self.assertTrue(isinstance(s2_a2, ak.Series))
        self.assertListEqual(s2_a2.index.to_list(), [3,3])
        self.assertListEqual(s2_a2.values.to_list(), ['DE', 'Z'])

        s3 = ak.Series(index=floats, data=ints)
        with self.assertRaises(TypeError):
            s3.loc[1]
        with self.assertRaises(TypeError):
            s3.loc['A']
        s3_a1 = s3.loc[0.0]
        self.assertTrue(isinstance(s3_a1, ak.Series))
        self.assertListEqual(s3_a1.index.to_list(), [0.0])
        self.assertListEqual(s3_a1.values.to_list(), [0])

        s3_a2 = s3.loc[1.5]
        self.assertTrue(isinstance(s3_a2, ak.Series))
        self.assertListEqual(s3_a2.index.to_list(), [1.5,1.5])
        self.assertListEqual(s3_a2.values.to_list(), [1,7])
        pass

    def test_iloc(self):
        pass

    def test_at(self):
        pass

    def test_iat(self):
        pass